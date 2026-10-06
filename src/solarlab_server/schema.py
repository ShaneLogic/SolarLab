"""Exact prepared metadata delivery; no store or model operations."""

from fastapi import APIRouter
from starlette.responses import Response

from solarlab.config.schema import configuration_schema_representation

router = APIRouter()


@router.get("/configuration-schema")
def configuration_schema() -> Response:
    data, digest = configuration_schema_representation()
    return Response(data, media_type="application/json", headers={"ETag": f'"{digest}"'})
