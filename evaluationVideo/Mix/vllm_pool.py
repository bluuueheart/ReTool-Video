"""Free-memory-aware vLLM replica pool + worker count planner.

GPUs are sometimes shared with other jobs.  Instead of assuming "one worker per
card", this module probes the free memory of every candidate GPU and decides,
per card, (a) whether a vLLM replica fits at all, (b) how much memory the
replica may take (``--gpu-memory-utilization``) and (c) how many client workers
can run next to it (each worker keeps small embedding / ASR models on the GPU).

CLI::

    python vllm_pool.py plan   --devices 0      # dry-run, prints the plan
    python vllm_pool.py up     --devices 0      # plan + launch + wait healthy
    python vllm_pool.py status
    python vllm_pool.py down
"""

from __future__ import annotations

import argparse
import atexit
import json
import logging
import math
import os
import shutil
import signal
import socket
import subprocess
import time
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

LOGGER = logging.getLogger("vllm_pool")

GIB = 1024.0
DEFAULT_ROOT = Path(os.environ.get("TOOLSTAR_VLLM_POOL_ROOT") or (Path(__file__).resolve().parents[2] / "local_runs" / "vllm_pool"))
DEFAULT_MODEL = os.environ.get("TOOLSTAR_VLLM_MODEL_PATH") or str(Path(os.environ.get("MODEL_BASE_PATH", "./models")) / "Qwen/Qwen3.5-9B")
# Cap on client workers per GPU. The environment can request a lower cap.
HARD_MAX_WORKERS_PER_GPU = 6
TOKENS_PER_KV_GIB = 8000  # measured on Qwen3.5-9B (hybrid attention): 36.75 GiB -> 300k tokens


@dataclass
class GpuInfo:
    index: str
    total_gib: float
    used_gib: float
    free_gib: float


@dataclass
class Replica:
    device: str
    port: int
    util: float
    workers: int
    max_model_len: int
    kv_gib: float
    pid: int = 0
    owned: bool = False
    log: str = ""
    process_start_ticks: Optional[int] = None
    boot_id: Optional[str] = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"


@dataclass
class PlanParams:
    weights_gib: float = 18.0       # bf16 9B weights
    overhead_gib: float = 6.0       # activations, cuda graphs, cuda context
    kv_min_gib: float = 12.0        # below this the card is skipped
    kv_target_gib: float = 40.0     # replica never takes more than weights+overhead+this
    aux_gib_per_worker: float = 4.0  # embedding + ASR + CUDA context of one client worker
    margin_gib: float = 3.0         # always left free for other processes
    max_workers_per_gpu: int = HARD_MAX_WORKERS_PER_GPU
    max_total_workers: int = 0      # 0 = unlimited
    max_model_len_cap: int = 262144
    max_num_seqs: int = 32

    @classmethod
    def from_env(cls) -> "PlanParams":
        p = cls()
        for key, caster in (
            ("weights_gib", float), ("overhead_gib", float), ("kv_min_gib", float),
            ("kv_target_gib", float), ("aux_gib_per_worker", float), ("margin_gib", float),
            ("max_workers_per_gpu", int), ("max_total_workers", int),
            ("max_model_len_cap", int), ("max_num_seqs", int),
        ):
            raw = os.environ.get(f"TOOLSTAR_VLLM_{key.upper()}")
            if raw not in (None, ""):
                setattr(p, key, caster(raw))
        p.max_workers_per_gpu = max(1, min(int(p.max_workers_per_gpu), HARD_MAX_WORKERS_PER_GPU))
        return p


# --------------------------------------------------------------------- probing
def probe_gpus(candidate_ids: Optional[Sequence[str]] = None) -> List[GpuInfo]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.total,memory.used,memory.free",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=30, check=True,
    ).stdout
    wanted = {str(c).strip() for c in candidate_ids} if candidate_ids is not None else None
    gpus: List[GpuInfo] = []
    for line in out.strip().splitlines():
        idx, total, used, free = [x.strip() for x in line.split(",")]
        if wanted is not None and idx not in wanted:
            continue
        gpus.append(GpuInfo(idx, float(total) / GIB, float(used) / GIB, float(free) / GIB))
    return gpus


def plan_replicas(
    gpus: Sequence[GpuInfo],
    params: PlanParams,
    *,
    ports: Optional[Dict[str, int]] = None,
) -> List[Replica]:
    """Pure function: free memory per card -> replica/worker layout."""
    need_min = params.weights_gib + params.overhead_gib + params.kv_min_gib
    cap = params.weights_gib + params.overhead_gib + params.kv_target_gib
    replicas: List[Replica] = []
    for gpu in sorted(gpus, key=lambda g: -g.free_gib):
        usable = gpu.free_gib - params.margin_gib
        workers = int(math.floor((usable - need_min) / max(0.5, params.aux_gib_per_worker)))
        workers = min(workers, params.max_workers_per_gpu)
        if workers < 1:
            LOGGER.info("GPU %s skipped: free=%.1f GiB < needed %.1f GiB (+1 worker)",
                        gpu.index, gpu.free_gib, need_min + params.aux_gib_per_worker)
            continue
        replica_gib = min(usable - workers * params.aux_gib_per_worker, cap)
        kv_gib = replica_gib - params.weights_gib - params.overhead_gib
        max_len = int(min(params.max_model_len_cap, (kv_gib * TOKENS_PER_KV_GIB) // 1024 * 1024))
        max_len = max(32768, max_len)
        port = (ports or {}).get(gpu.index, 8100 + int(gpu.index))
        replicas.append(Replica(
            device=gpu.index, port=port, util=round(replica_gib / gpu.total_gib, 4),
            workers=workers, max_model_len=max_len, kv_gib=round(kv_gib, 1),
        ))
    replicas.sort(key=lambda r: int(r.device))
    if params.max_total_workers > 0:
        budget = params.max_total_workers
        for r in sorted(replicas, key=lambda r: -r.workers):
            r.workers = min(r.workers, max(0, budget))
            budget -= r.workers
        replicas = [r for r in replicas if r.workers > 0]
    return replicas


# ------------------------------------------------------------------- lifecycle
def _healthy(url: str, timeout: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/models", timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def _port_free(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) != 0


def plan_path(root: Path = DEFAULT_ROOT) -> Path:
    return root / "plan.json"


def load_plan(path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    path = Path(path or os.environ.get("TOOLSTAR_VLLM_PLAN") or plan_path())
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def endpoint_for_device(device: str, plan: Optional[Dict[str, Any]] = None) -> str:
    """Home replica URL for the worker bound to ``device`` ("" when none)."""
    plan = plan or load_plan()
    if not plan:
        return ""
    for rep in plan.get("replicas", []):
        if str(rep.get("device")) == str(device):
            return f"http://127.0.0.1:{int(rep['port'])}/v1"
    return ""


def _die_with_parent() -> None:
    """Child gets SIGTERM if the launching process dies for any reason (even SIGKILL)."""
    try:
        import ctypes

        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
    except Exception:
        pass


def _launch(rep: Replica, *, model: str, served: str, root: Path, die_with_parent: bool = True) -> None:
    root.mkdir(parents=True, exist_ok=True)
    log = root / f"replica_gpu{rep.device}.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = rep.device
    if "VLLM_CACHE_ROOT" not in env and env.get("XDG_CACHE_HOME"):
        env["VLLM_CACHE_ROOT"] = str(Path(env["XDG_CACHE_HOME"]) / "vllm")
    vllm_bin_dir = os.environ.get("TOOLSTAR_VLLM_BIN")
    if vllm_bin_dir:
        env["PATH"] = str(Path(vllm_bin_dir).expanduser().resolve()) + os.pathsep + env.get("PATH", "")
    executable = shutil.which("vllm", path=env.get("PATH"))
    if executable is None:
        raise RuntimeError("vLLM executable not found. Install vLLM or set TOOLSTAR_VLLM_BIN.")
    cmd = [
        executable, "serve", model,
        "--served-model-name", served, "--host", "127.0.0.1", "--port", str(rep.port),
        "--max-model-len", str(rep.max_model_len),
        "--gpu-memory-utilization", str(rep.util),
        "--kv-cache-memory-bytes", str(int(rep.kv_gib * 1024 ** 3)),
        "--max-num-seqs", str(PlanParams.from_env().max_num_seqs),
        "--limit-mm-per-prompt", json.dumps({"image": 64, "video": 4}),
        "--allowed-local-media-path", os.environ.get("TOOLSTAR_VLLM_MEDIA_ROOT", str(Path.cwd())),
        "--default-chat-template-kwargs", json.dumps({"enable_thinking": False}),
        # Model card: with num_frames=-1 the request's mm_processor_kwargs (fps, do_sample_frames) decide the sampling.
        "--media-io-kwargs", json.dumps({"video": {"num_frames": -1}}),
        # The mm processor cache is keyed on the API-server side and breaks when the
        # engine cache misses (vLLM 0.19 AssertionError); preprocessing cost is small.
        "--mm-processor-cache-gb", "0",
    ]
    if os.environ.get("TOOLSTAR_VLLM_REASONING_PARSER", "1") != "0":
        # Thinking-mode replies: reasoning goes to a separate field, `content` keeps the answer only.
        cmd += ["--reasoning-parser", "qwen3"]
    if os.environ.get("TOOLSTAR_VLLM_PREFIX_CACHING", "1") != "0":
        # Planner turns reuse the conversation prefix.
        cmd.append("--enable-prefix-caching")
    lora = str(os.environ.get("TOOLSTAR_VLLM_LORA_MODULES", "") or "").strip()
    if lora:
        # "name=/path/to/adapter[,name2=path2]": adapters are served next to the base model (request model=name picks one,
        # model=<served name> is the untouched base). Unset keeps the old command line.
        cmd += ["--enable-lora", "--max-lora-rank", os.environ.get("TOOLSTAR_VLLM_MAX_LORA_RANK", "64"), "--max-loras", "1",
                "--lora-modules", *lora.split(",")]
    with log.open("ab") as handle:
        proc = subprocess.Popen(cmd, env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True,
                                preexec_fn=_die_with_parent if die_with_parent else None)
    rep.pid, rep.owned, rep.log = proc.pid, True, str(log)
    rep.process_start_ticks = _process_start_ticks(rep.pid)
    rep.boot_id = _boot_id()


def _boot_id() -> Optional[str]:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return None


def _process_start_ticks(pid: int) -> Optional[int]:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        fields = stat.rsplit(")", 1)[1].split()
        if fields[0] in {"Z", "X"}:
            return None
        return int(fields[19])
    except (OSError, IndexError, ValueError):
        return None


def _matches_process(replica: Replica) -> bool:
    return (
        replica.pid > 0
        and replica.boot_id is not None
        and replica.boot_id == _boot_id()
        and replica.process_start_ticks is not None
        and replica.process_start_ticks == _process_start_ticks(replica.pid)
    )


def _stop_replica(replica: Replica) -> None:
    if not _matches_process(replica):
        return
    try:
        if os.getpgid(replica.pid) == replica.pid:
            os.killpg(replica.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass


def _write_plan(root: Path, model: str, served: str, replicas: Sequence[Replica]) -> Dict[str, Any]:
    plan = {
        "created": time.strftime("%F %T"),
        "model_path": model,
        "served_name": served,
        "replicas": [asdict(r) for r in replicas],
        "total_workers": sum(r.workers for r in replicas),
        "base_urls": ",".join(r.url for r in replicas),
    }
    temporary_path = root / f".plan-{os.getpid()}.json"
    temporary_path.write_text(json.dumps(plan, indent=2))
    temporary_path.replace(plan_path(root))
    return plan


def ensure_pool(
    devices: Sequence[str],
    *,
    root: Path = DEFAULT_ROOT,
    model: str = DEFAULT_MODEL,
    served: str = "qwen35-9b",
    params: Optional[PlanParams] = None,
    startup_timeout: float = 1200.0,
    stop_on_exit: bool = True,
    needed_workers: Optional[int] = None,
) -> Dict[str, Any]:
    """Reuse healthy replicas, launch the missing ones sized by free memory, return the plan.

    ``needed_workers``: only bring up as many replicas as it takes to host that many
    concurrent workers (cards are taken in the given order); the rest stay free.
    """
    params = params or PlanParams.from_env()
    root.mkdir(parents=True, exist_ok=True)
    devices = [str(d) for d in devices]
    existing = load_plan(plan_path(root)) or {}
    reused: List[Replica] = []
    for item in existing.get("replicas", []):
        rep = Replica(**{k: item[k] for k in Replica.__dataclass_fields__ if k in item})
        if not _matches_process(rep):
            continue
        if (existing.get("model_path"), existing.get("served_name")) != (model, served):
            raise RuntimeError(f"Pool at {root} is running a different model. Stop it or use another pool root.")
        if rep.device in devices and _healthy(rep.url):
            rep.owned = False  # someone else (an earlier run) owns it; do not stop it on exit
            reused.append(rep)
            LOGGER.info("Reusing healthy vLLM replica on GPU %s (%s)", rep.device, rep.url)
    reused_devices = {r.device for r in reused}

    excluded: set = set()
    launched: List[Replica] = []
    for _ in range(3):  # replan if a launch fails because someone grabbed the memory
        gpus = [g for g in probe_gpus([d for d in devices if d not in reused_devices | excluded])]
        ports = {g.index: 8100 + int(g.index) for g in gpus}
        for g in gpus:
            port = next((port for port in (ports[g.index], ports[g.index] + 100) if _port_free(port)), None)
            if port is None:
                raise RuntimeError(f"No available vLLM port for GPU {g.index}.")
            ports[g.index] = port
        todo = plan_replicas(gpus, params, ports=ports)
        if needed_workers:
            have = sum(r.workers for r in reused + launched)
            order = {d: i for i, d in enumerate(devices)}
            chosen: List[Replica] = []
            for t in sorted(todo, key=lambda t: order.get(t.device, 1 << 30)):
                if have >= int(needed_workers):
                    break
                chosen.append(t)
                have += t.workers
            todo = chosen
        for r in launched:
            if r.device in {t.device for t in todo}:
                todo = [t for t in todo if t.device != r.device]
        if not todo:
            break
        for rep in todo:
            LOGGER.info("Launching vLLM replica GPU=%s port=%d util=%.3f workers=%d max_len=%d kv=%.1fGiB",
                        rep.device, rep.port, rep.util, rep.workers, rep.max_model_len, rep.kv_gib)
            _launch(rep, model=model, served=served, root=root, die_with_parent=stop_on_exit)
            _write_plan(root, model, served, reused + launched + [rep])  # never orphan a server
        deadline = time.time() + startup_timeout
        failed: List[Replica] = []
        pending = list(todo)
        while pending and time.time() < deadline:
            for rep in list(pending):
                if not _matches_process(rep):
                    pending.remove(rep)
                    failed.append(rep)
                elif _healthy(rep.url):
                    pending.remove(rep)
                    launched.append(rep)
                    LOGGER.info("vLLM replica GPU=%s healthy", rep.device)
            time.sleep(3)
        failed.extend(pending)
        for rep in failed:
            LOGGER.warning("vLLM replica GPU=%s failed to start (see %s); excluding the card", rep.device, rep.log)
            _stop_replica(rep)
            excluded.add(rep.device)
        if not failed:
            break

    replicas = sorted(reused + launched, key=lambda r: int(r.device))
    if not replicas:
        raise RuntimeError("No GPU has enough free memory for a vLLM replica (see vllm_pool plan).")
    plan = _write_plan(root, model, served, replicas)
    if stop_on_exit and launched:
        def stop_launched_replicas():
            for replica in launched:
                _stop_replica(replica)

        atexit.register(stop_launched_replicas)
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous = signal.getsignal(sig)

            def _handler(signum, frame, _prev=previous):
                stop_launched_replicas()
                signal.signal(signum, _prev if callable(_prev) else signal.SIG_DFL)
                os.kill(os.getpid(), signum)

            signal.signal(sig, _handler)
    return plan


def stop_pool(root: Path = DEFAULT_ROOT) -> None:
    plan = load_plan(plan_path(root)) or {}
    for item in plan.get("replicas", []):
        replica = Replica(**item)
        if replica.pid and (replica.process_start_ticks is None or replica.boot_id is None):
            LOGGER.warning("Skipping unverifiable PID %s from legacy pool plan.", replica.pid)
        _stop_replica(replica)
    try:
        plan_path(root).unlink()
    except FileNotFoundError:
        pass


def expand_workers(plan: Dict[str, Any], limit: Optional[int] = None) -> List[str]:
    """Device id per worker slot, fill-first: use up one card's workers, then open the next.

    A card's ``workers`` already reflects its free memory (other processes on the card shrink it),
    so "no headroom" simply means that card contributes fewer or no slots. Cards are used in
    ascending device order.
    """
    slots: List[str] = []
    per_card: List[List[str]] = []
    for rep in sorted(plan["replicas"], key=lambda r: int(r["device"])):
        card = [str(rep["device"])] * max(0, min(int(rep["workers"]), HARD_MAX_WORKERS_PER_GPU))
        per_card.append(card)
        slots.extend(card)
    if os.environ.get("TOOLSTAR_VLLM_SPREAD_WORKERS") == "1":
        # Cards are already up (shared pool): hand slots out round-robin so a small batch uses every card.
        slots = [card[i] for i in range(max((len(c) for c in per_card), default=0)) for card in per_card if i < len(card)]
    return slots[:limit] if limit else slots


def workers_for_device(device: str, plan: Optional[Dict[str, Any]] = None) -> int:
    """Concurrent-worker budget of one card from the plan (0 = no replica / unknown)."""
    plan = plan or load_plan()
    for rep in (plan or {}).get("replicas", []):
        if str(rep.get("device")) == str(device):
            return max(0, min(int(rep.get("workers") or 0), HARD_MAX_WORKERS_PER_GPU))
    return 0


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("plan", "up", "status", "down"))
    parser.add_argument("--devices", default=os.environ.get("CUDA_VISIBLE_DEVICES", os.environ.get("TOOLSTAR_ALLOWED_GPUS")))
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    args = parser.parse_args()
    if args.command in {"plan", "up"}:
        devices = (
            [gpu.index for gpu in probe_gpus()]
            if args.devices is None
            else [device.strip() for device in args.devices.split(",") if device.strip() and device.strip() != "-1"]
        )
        if not devices:
            parser.error("No GPUs selected. Set --devices or CUDA_VISIBLE_DEVICES.")
    root = Path(args.root)
    if args.command == "plan":
        gpus = probe_gpus(devices)
        for g in gpus:
            print(f"GPU {g.index}: total={g.total_gib:.0f} used={g.used_gib:.1f} free={g.free_gib:.1f} GiB")
        reps = plan_replicas(gpus, PlanParams.from_env())
        print(json.dumps([asdict(r) for r in reps], indent=2))
        print("total workers:", sum(r.workers for r in reps))
    elif args.command == "up":
        plan = ensure_pool(devices, root=root, stop_on_exit=False)
        print(json.dumps(plan, indent=2))
    elif args.command == "status":
        plan = load_plan(plan_path(root)) or {}
        for rep in plan.get("replicas", []):
            replica = Replica(**rep)
            print(replica.device, replica.port, "healthy" if _matches_process(replica) and _healthy(replica.url) else "DOWN")
    else:
        stop_pool(root)


if __name__ == "__main__":
    main()
