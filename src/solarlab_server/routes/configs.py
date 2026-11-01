"""Read explicitly supplied sources and retain the existing typed previews."""
from fastapi import APIRouter, HTTPException, Request
from starlette.responses import Response

from solarlab_server.configuration import router as preview_router

router = APIRouter()
router.include_router(preview_router)


@router.get("/configs")
def configs(request: Request) -> dict:
    context = request.app.state.configuration_preview
    if context is None:
        raise HTTPException(503, "trusted configuration context is unavailable")
    return {"sources": [{"id": s.id, "sha256": s.sha256, "bytes": len(s.content)} for s in context.sources],
            "default_catalog_sha256": context.defaults.content_sha256}


@router.get("/configs/{source_id:path}")
def source(request: Request, source_id: str) -> Response:
    context = request.app.state.configuration_preview
    if context is None:
        raise HTTPException(503, "trusted configuration context is unavailable")
    item = next((s for s in context.sources if s.id == source_id), None)
    if item is None:
        raise HTTPException(404, "unknown supplied configuration source")
    return Response(item.content, media_type="application/octet-stream", headers={"ETag": '"' + item.sha256 + '"'})
