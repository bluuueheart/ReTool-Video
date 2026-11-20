"""Complete tool results from one resolver session, for the parent planner.

The resolver's `Return` value is its own conclusion (often just an option letter). It never replaces the
tool outputs: answers, evidence, uncertainty and model reasoning must reach the parent unchanged.
"""
import json
from typing import Any, Dict, List

# Control or interface-discovery steps. Tool interface metadata is not video evidence.
_NON_EVIDENCE_ACTIONS = {"resolver_return", "return", "finish", "search", "tool_search", "system", "protocol_error", "assistant_note"}
_TREE_ACTIONS = {"execute_plan", "resolver_turn"}


def _strip_internal(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _strip_internal(v) for k, v in value.items() if not str(k).startswith("__")}
    if isinstance(value, list):
        return [_strip_internal(v) for v in value]
    return value


def _tree_leaves(tree: Any, step_id: int) -> List[Dict[str, Any]]:
    leaves = []
    for entry in (tree.get("execution_tree") or []) if isinstance(tree, dict) else []:
        if not isinstance(entry, dict) or not entry.get("tool"):
            continue  # grouping nodes are not tool calls
        record: Dict[str, Any] = {"tool": str(entry["tool"]), "status": str(entry.get("status") or ""), "step": step_id}
        if entry.get("id"):
            record["node_id"] = entry["id"]
        result = entry.get("result")
        record.update(result if isinstance(result, dict) else {"result": result})
        record["tool"], record["status"] = str(entry["tool"]), str(entry.get("status") or record.get("status") or "")
        leaves.append(_strip_internal(record))
    return leaves


def collect_executed_results(steps: List[Any]) -> List[Dict[str, Any]]:
    """Return every concrete tool result, in order, exactly as the resolver observed it."""
    collected: List[Dict[str, Any]] = []
    for step in steps or []:
        action = str(getattr(step, "action", "") or "").strip()
        if action.lower() in _NON_EVIDENCE_ACTIONS:
            continue
        for item in getattr(step, "observation_items", None) or []:
            if not isinstance(item, dict):
                continue
            tree = item.get("ActionResults") if isinstance(item.get("ActionResults"), dict) else item
            if action.lower() in _TREE_ACTIONS or "execution_tree" in tree:
                collected.extend(_tree_leaves(tree, step.step_id))
                continue
            tool = str(item.get("tool") or action).strip()
            if tool.lower() in _NON_EVIDENCE_ACTIONS:
                continue
            record = dict(item)
            result = record.get("result")
            if isinstance(result, str):
                try:
                    parsed = json.loads(result)
                except ValueError:
                    parsed = None
                if isinstance(parsed, dict):  # A single-call observation stored as serialized JSON.
                    record = {**parsed, **{k: v for k, v in record.items() if k != "result"}}
            record["tool"], record["step"] = tool, step.step_id
            collected.append(_strip_internal(record))
    return collected


def build_planner_payload(executed_results: List[Dict[str, Any]], resolver_return: Any, *, task_status=None) -> Any:
    """One result: the tool's own payload. Several: all of them. The resolver's note is kept separately."""
    if not executed_results:
        return resolver_return
    if len(executed_results) == 1:
        payload = {k: v for k, v in executed_results[0].items() if k not in {"step"}}
        payload.pop("tool", None)
    else:
        payload = {"items": [{k: v for k, v in r.items() if k != "step"} for r in executed_results]}
        statuses = [str(r.get("status", "")).lower() for r in executed_results]
        payload["status"] = "success" if all(s == "success" for s in statuses) else "error"
        payload["execution_status"] = "partial_success" if "success" in statuses and payload["status"] == "error" else payload["status"]
        payload["successful_items"] = statuses.count("success")
        payload["failed_items"] = len(statuses) - statuses.count("success")
    if task_status is not None:
        payload["status"] = task_status
        payload["task_status"] = task_status
    # Keep structured absolute records accessible without reconstructing them from raw clip prose.
    if len(executed_results) > 1:
        events = []
        for result in executed_results:
            if result.get("status") != "success":
                continue
            for event in result.get("evidence_events") or []:
                if isinstance(event, dict) and event.get("timestamp_basis") == "absolute":
                    record = dict(event)
                    record.setdefault("source_tool", result.get("tool"))
                    if result.get("node_id"):
                        record.setdefault("source_node_id", result["node_id"])
                    events.append(record)
        if events:
            payload["evidence_events"] = events
    stripped = [{k: v for k, v in r.items() if k not in {"step", "tool", "status"}} for r in executed_results]
    if resolver_return not in (None, "") and not any(resolver_return == r for r in stripped):
        payload["resolver_return"] = resolver_return  # Skip an exact echo of a tool result.
    return payload
