"""Separate control flow, capability calls and explicitly tracked data dependencies."""
import re
from collections import defaultdict

DISCOVERY_TOOLS = frozenset({"Tool_Search"})
CONTROL_ACTIONS = frozenset({"Finish", "Return", "Resolver_Return", "Search", "System", "Assistant_Note",
                             "Protocol_Error", "Parallel", "Execute_Plan", "Resolver_Turn"}) | DISCOVERY_TOOLS


def action_tool_names(action, args):
    if action == "Parallel":
        return [name for call in args.get("batch", [])
                for name in action_tool_names(call.get("name", ""), call.get("args") or {})]
    if action in {"Execute_Plan", "Resolver_Turn"}:
        def visit(nodes):
            for node in nodes:
                if not isinstance(node, dict):
                    continue
                if isinstance(node.get("children"), list):
                    yield from visit(node["children"])
                elif node.get("tool") not in CONTROL_ACTIONS and node.get("tool"):
                    yield node["tool"]
        return list(visit(args.get("nodes", [])))
    return [action] if action and action not in CONTROL_ACTIONS else []


def iter_result_references(value):
    if isinstance(value, str) and value.startswith("$") and len(value) > 1:
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from iter_result_references(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_result_references(child)


def explicit_dependency_metrics(trajectories):
    """Track resolved $node/$last_tool references; literal rewrites are not inferred.

    Unknown producers (e.g. external inputs/group aliases) are reported explicitly.
    A reported depth is the longest tracked chain, not a claim of complete provenance.
    """
    edges, untracked, depths = [], [], {}
    calls = [e for t in trajectories for e in t.get("tool_execution_events", [])
             if e.get("executed") and e["tool"] not in CONTROL_ACTIONS]
    available = all("result_references" in e for e in calls)
    for trajectory in trajectories:
        aliases = {}
        iterations = defaultdict(list)
        for event in trajectory.get("tool_execution_events", []):
            if event["tool"] not in CONTROL_ACTIONS:
                iterations[event.get("iteration")].append(event)
        for events in iterations.values():
            parallel_snapshots = {}
            for event in events:
                event_id = event["event_id"]
                if not event.get("executed"):
                    # A host-side failure may replace a previous result; do not invent an edge to it.
                    name = event["tool"]
                    normalized = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
                    for label in {name, normalized}:
                        for alias in (label, label + "_result", "last_" + label + "_result"):
                            aliases.pop(alias, None)
                    aliases.pop(event.get("plan_node_id"), None)
                    continue
                group = event.get("dispatch_group") if event.get("dispatch_mode") == "parallel" else None
                inputs = parallel_snapshots.setdefault(group, dict(aliases)) if group else aliases
                producers = set()
                for reference, resolved in event.get("result_references", {}).items():
                    if not resolved:
                        continue
                    root = reference[1:].split(".", 1)[0]
                    # Generic envelopes/group outputs can have several producers and host errors.
                    previous = inputs.get(root) if root != "last_result" else None
                    if previous is None:
                        untracked.append({"event_id": event_id, "reference": reference})
                    else:
                        producers.update(previous)
                depths[event_id] = 1 + max((depths[p] for p in producers), default=0)
                edges.extend({"producer": p, "consumer": event_id} for p in sorted(producers))
                name = event["tool"]
                normalized = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
                for label in {name, normalized}:
                    for alias in (label, label + "_result", "last_" + label + "_result"):
                        aliases[alias] = {event_id}
                if event.get("plan_node_id"):
                    aliases[event["plan_node_id"]] = {event_id}
    return {"explicit_dependency_tracking_available": available,
            "tracked_explicit_dependency_depth": max(depths.values(), default=0) if available else None,
            "explicit_dependency_edges": edges, "untracked_result_references": untracked}
