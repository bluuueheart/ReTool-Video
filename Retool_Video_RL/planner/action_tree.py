"""Execute a model-provided hierarchy; model turns and tree levels are separate."""
from typing import Any, Callable


def execute_action_tree(nodes: list, execute: Callable, store: Callable, max_depth: int) -> dict:
    trace = []

    def visit(node: dict, level: int) -> dict:
        if level > max_depth:
            raise ValueError(f"Action tree exceeds existing max_recursion_depth={max_depth}")
        if not isinstance(node, dict):
            raise ValueError("Each action-tree node must be an object")
        node_id = str(node.get("id") or "")
        children = node.get("children")
        if isinstance(children, list) and node.get("tool"):
            value = {"error": "A node cannot contain both tool and children. Use a children-only group and put each tool call in a leaf inside it."}
            status, executed = "error", False
        elif isinstance(children, list):
            outcomes = [visit(child, level + 1) for child in children]
            value = {"items": outcomes}
            status = "success" if outcomes and all(c["status"] == "success" for c in outcomes) else "error"
            value["status"] = status
            executed = False  # A grouping node is not a tool/model call.
        elif not node.get("tool"):
            value = {"error": "A leaf needs tool; a grouping node needs children. Put discovery requests in top-level Search."}
            status, executed = "error", False
        else:
            tool = str(node.get("tool") or "")
            try:
                args = dict(node.get("params") or {})
                if node.get("description"):
                    args["__description__"] = node["description"]
                if "search_id" in node:
                    args["__search_id__"] = node["search_id"]
                value, status, executed = execute(tool, args, level, node_id)
            except Exception as exc:
                value, status, executed = {"error": str(exc)}, "error", False
        if node_id:
            store(node_id, value)
        entry = {"id": node_id, "plan_depth": level, "tool": node.get("tool"),
                 "status": status, "executed": executed, "result": value}
        if "search_id" in node:
            entry["search_id"] = node["search_id"]
        trace.append(entry)
        return entry

    outcomes = [visit(node, 1) for node in nodes]
    return {"items": outcomes, "status": "success" if outcomes and all(n["status"] == "success" for n in outcomes) else "error",
            "execution_tree": trace, "max_plan_depth": max((n["plan_depth"] for n in trace), default=0)}
