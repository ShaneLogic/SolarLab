"""Read actual immutable resource metadata and source bytes, without discovery."""
from fastapi import APIRouter, HTTPException, Request
from starlette.responses import Response

router = APIRouter()


@router.get("/materials")
def materials(request: Request) -> dict:
    context = request.app.state.configuration_preview
    if context is None:
        raise HTTPException(503, "trusted material context is unavailable")
    return {"resource_library_sha256": context.resources.content_sha256,
        "resources": [{"name": t.name, "kind": t.kind, "sha256": t.content_sha256,
            "source_sha256": t.source.sha256, "shape": t.shape, "columns": t.columns, "units": t.units}
            for t in context.resources.tables]}


@router.get("/materials/{name}")
def material(request: Request, name: str) -> Response:
    context = request.app.state.configuration_preview
    if context is None:
        raise HTTPException(503, "trusted material context is unavailable")
    table = next((t for t in context.resources.tables if t.name == name), None)
    if table is None:
        raise HTTPException(404, "unknown supplied material resource")
    return Response(table.source.content, media_type="application/octet-stream",
                    headers={"ETag": '"' + table.source.sha256 + '"'})
