"""Internal routes for inmem runtime (testing/debugging only)."""

import asyncio

from langgraph_runtime_inmem.database import connect


def get_internal_routes():
    from langgraph_api.config import MIGRATIONS_PATH

    try:
        from langgraph_api.middleware import http_logger

        http_logger.PATHS_IGNORE.add("/internal/truncate")
    except ImportError:
        pass

    if "__inmem" not in MIGRATIONS_PATH:
        # not in a testing mode.
        return []
    from langgraph_api.route import ApiRequest, ApiResponse, ApiRoute

    async def truncate(request: ApiRequest):
        """Truncate all inmem data (for testing)."""
        from langgraph_runtime.checkpoint import Checkpointer

        await asyncio.to_thread(Checkpointer().clear)
        async with connect() as conn:
            await asyncio.to_thread(conn.clear)
        return ApiResponse({"ok": True})

    async def debug_get_raw_thread(request: ApiRequest):
        """Return raw thread from store without decryption (for testing)."""
        thread_id = request.path_params["thread_id"]
        async with connect() as conn:
            for thread in conn.store["threads"]:
                if str(thread["thread_id"]) == thread_id:
                    return ApiResponse(thread)
        return ApiResponse({"error": "not found"}, status_code=404)

    return [
        ApiRoute("/internal/truncate", truncate, methods=["POST"]),
        ApiRoute(
            "/internal/debug/thread/{thread_id}", debug_get_raw_thread, methods=["GET"]
        ),
    ]
