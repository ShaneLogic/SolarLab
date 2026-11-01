"""Actual catalog compositions shared by the base oracle and V3 regressions."""
from dataclasses import replace


METHODS = ("ionic_base", "other_base", "constructor")
KINDS = ("legacy", "ion_free", "ionic_transient", "two_dimensional")


def companions(implementation):
    import job_route_fixtures
    from examples.research_1d.ionic_transient_service import installed_catalog as ionic_transient
    from examples.research_1d.transient_service import installed_catalog as ion_free
    from examples.research_2d.http_steady import installed_catalog as two_dimensional

    head = implementation.build_source_commit
    return {"legacy": job_route_fixtures.catalog()[0],
            "ion_free": ion_free(implementation, head),
            "ionic_transient": ionic_transient(implementation, head),
            "two_dimensional": two_dimensional(implementation, head)}


def compose(ionic, other, method, reverse, *, wall_seconds=None):
    from solarlab.experiments.registry import ExperimentRegistry
    from solarlab_server.job_service import ExecutionCatalog

    catalogs = (other, ionic) if reverse else (ionic, other)
    sources = {source.id: source for catalog in catalogs for source in catalog.sources}
    registry = ExperimentRegistry(tuple(d for catalog in catalogs for d in catalog.registry.definitions))
    ceiling = replace(ionic.ceiling, wall_seconds=(
        max(ionic.ceiling.wall_seconds, other.ceiling.wall_seconds) if wall_seconds is None else wall_seconds))
    common = dict(registry=registry, models=other.models, expected_registry_sha256=registry.content_sha256,
        expected_model_sha256=other.models.content_sha256, sources=tuple(sources.values()),
        registrations=tuple(r for catalog in catalogs for r in catalog.registrations), ceiling=ceiling)
    if method == "constructor":
        return ExecutionCatalog(**common)
    return replace(ionic if method == "ionic_base" else other, **common)


def submissions(kind):
    import job_route_fixtures
    from examples.research_1d.ionic_jv_service import example_submission as ionic_jv
    from examples.research_1d.ionic_transient_service import example_submission as ionic_transient
    from examples.research_1d.transient_service import example_submission as transient
    from examples.research_2d.http_steady import example_submission as two_dimensional

    bodies = [ionic_jv()]
    if kind == "legacy":
        bodies.append(job_route_fixtures.request())
    elif kind == "ion_free":
        bodies.append(transient())
        steady = ionic_jv()
        steady.update(experiment_id="steady_jv", input={"device_id": "research_1d", "voltage_V": [0., .05, .1],
            "intervals_per_layer": 12, "generation_scale": 1., "incident_power_W_m2": None})
        bodies.append(steady)
    else:
        bodies.append(ionic_transient() if kind == "ionic_transient" else two_dimensional())
    walls = (30., 60., 61., 128.) if kind == "ion_free" else (30., 31., 128.)
    return [(f"{kind}/{body['experiment_id']}/{int(wall)}",
             {**body, "limits": {**body["limits"], "wall_seconds": wall}})
            for body in bodies for wall in walls]


def admit(catalog, body):
    from solarlab_server.job_service import JobSubmission
    try:
        spec, _ = catalog.prepare(JobSubmission.model_validate(body))
    except ValueError as error:
        if not any(word in str(error) for word in ("wall", "ceiling", "requires <=30s")):
            raise
        return False
    assert spec.limits.wall_seconds == body["limits"]["wall_seconds"]
    return True
