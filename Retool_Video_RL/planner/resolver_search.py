"""Top-level resolver search requests; each request retains its own top five."""
from concurrent.futures import ThreadPoolExecutor


def run_searches(requests, search, previous_ids):
    if not isinstance(requests, list) or not requests:
        raise ValueError("Search must be a non-empty list of requests")
    seen = set(previous_ids)
    for request in requests:
        if not isinstance(request, dict):
            raise ValueError("Each Search request must be an object")
        search_id = request.get("id")
        if not isinstance(search_id, str) or not search_id.strip() or search_id in seen:
            raise ValueError("Each Search needs a non-empty, session-unique id")
        seen.add(search_id)
        if not isinstance(request.get("intent_description"), str) or not request["intent_description"].strip():
            raise ValueError("Each Search needs intent_description")
        scopes = request.get("scopes")
        if not isinstance(scopes, list) or not scopes or any(not isinstance(s, str) for s in scopes):
            raise ValueError("Each Search needs a non-empty scopes list")

    def run(request):
        group = {"search_id": request["id"], "intent_description": request["intent_description"],
                 "scopes": request["scopes"], "candidates": []}
        try:
            group["candidates"] = search.run(intent_description=request["intent_description"],
                                              scopes=request["scopes"], top_k=5)
            group["status"] = "success" if group["candidates"] else "no_valid_result"
        except Exception as exc:
            group.update(status="error", error=str(exc))
        return group

    with ThreadPoolExecutor(max_workers=len(requests)) as pool:
        return list(pool.map(run, requests))
