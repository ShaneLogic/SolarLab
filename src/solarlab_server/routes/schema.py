"""Generated schemas and their content identities; no admission by schema alone."""
import hashlib

from fastapi import APIRouter
from starlette.responses import Response

from solarlab.experiments.protocol import protocol_schema, working_point_schema
from solarlab.experiments.results import array_schema, result_schema
from solarlab_server.job_service import JobSubmission, canonical
from solarlab_server.schema import router as configuration_schema_router

router = APIRouter()
router.include_router(configuration_schema_router)


@router.get("/schema")
def schemas() -> Response:
    data = canonical({"protocol": protocol_schema(), "working_point": working_point_schema(),
        "result": result_schema(), "array": array_schema(), "job_submission": JobSubmission.model_json_schema(),
        "scope": "schemas_only; Python consistency, capability and execution checks remain required"})
    return Response(data, media_type="application/json", headers={"ETag": '"' + hashlib.sha256(data).hexdigest() + '"'})
