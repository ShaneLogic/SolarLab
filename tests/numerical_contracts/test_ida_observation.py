"""Source preparation checks and separately admitted native boundary checks.

Source checks use IDA_OBSERVATION_SOURCE_ARCHIVE, an explicit pinned sdist.
Native checks require SOLARLAB_IDA_OBSERVATION_NATIVE_TEST=1 and the reviewed
patched wheel. They must not be counted as passed when the SDK is unavailable.
"""
from fractions import Fraction
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tarfile

import pytest


@pytest.fixture
def builder():
    path = Path(__file__).resolve().parents[2] / "scripts/benchmarks/ida_observation/build.py"
    spec = importlib.util.spec_from_file_location("ida_observation_builder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def archive():
    name = os.environ.get("IDA_OBSERVATION_SOURCE_ARCHIVE")
    if not name:
        pytest.skip("supply the explicit SHA-verified 1.1.3 source archive")
    return Path(name)


class TestSourcePreparation:
    def test_exact_patch_and_unchanged_native_solve_calls(self, builder, archive, tmp_path):
        result = builder.prepare(archive, tmp_path / "prepared")
        source = Path(result["source"])
        pins = builder.pins()
        for name, spec in pins["files"].items():
            assert builder.digest(source / name) == spec["patched_sha256"]
        code = (source / "src/sksundae/_cy_ida.pyx").read_text()
        with tarfile.open(archive) as tar:
            original = tar.extractfile("scikit_sundae-1.1.3/src/sksundae/_cy_ida.pyx").read().decode()
        calls = lambda text: [line.strip() for line in text.splitlines() if "flag = IDASolve(" in line]
        assert len(calls(code)) == 3
        assert calls(code) == calls(original)
        assert code.count("self._obs_after_step(observation_before,") == 3
        assert code.count("self._obs_busy = True") == 3
        assert "_OBSERVATION_BUILD_ID = None" in code
        ast.parse((source / "src/sksundae/ida/_solver.py").read_text())
        assert result["compiler_run"] is False

    def test_bound_build_identity_is_explicit(self, builder, archive, tmp_path):
        identity = "a" * 64
        result = builder.prepare(archive, tmp_path / "bound", identity)
        text = (Path(result["source"]) / "src/sksundae/_cy_ida.pyx").read_text()
        assert "_OBSERVATION_BUILD_ID = " + repr(identity) in text
        assert "_OBSERVATION_BUILD_ID = None" not in text
        assert result["build_recipe_sha256"] == identity

    def test_source_tampering_is_rejected(self, builder, archive, tmp_path):
        pins = builder.pins()
        source = tmp_path / "changed"
        with tarfile.open(archive) as tar:
            for name in pins["files"]:
                target = source / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(tar.extractfile("scikit_sundae-1.1.3/" + name).read())
        target = source / "src/sksundae/c_ida.pxd"
        target.write_bytes(target.read_bytes() + b"\n# changed\n")
        with pytest.raises(ValueError, match="upstream file identity"):
            builder.apply_patch(source, pins)

    def test_no_overwriting_an_existing_attempt(self, builder, archive, tmp_path):
        work = tmp_path / "attempt"
        work.mkdir()
        with pytest.raises(ValueError, match="reuse or overwrite"):
            builder.prepare(archive, work)

    @pytest.mark.parametrize("pin", ["", "0" * 64, None])
    def test_missing_or_wrong_external_plan_pin(self, builder, tmp_path, pin):
        path = tmp_path / "plan.json"
        path.write_text('{"schema":"one"}')
        with pytest.raises(ValueError, match="external identity"):
            builder.load_pinned(path, pin)

    def test_resealed_metadata_cannot_replace_external_pin(self, builder, tmp_path):
        path = tmp_path / "plan.json"
        path.write_text('{"compiler":"approved"}')
        external = builder.digest(path)
        assert builder.load_pinned(path, external) == {"compiler": "approved"}
        path.write_text(json.dumps({"compiler": "changed", "identity": hashlib.sha256(b"changed").hexdigest()}))
        with pytest.raises(ValueError, match="external identity"):
            builder.load_pinned(path, external)

    def test_missing_sdk_blocks_before_build_process(self, builder, tmp_path, monkeypatch):
        monkeypatch.setattr(builder.subprocess, "Popen", lambda *a, **k: pytest.fail("unexpected build process"))
        recipe = dict.fromkeys(["python", "python_sha256", "python_version", "cython", "compiler",
                              "platform_tools", "sdk_files", "private_files", "python_include",
                              "python_headers", "numpy_include", "numpy_headers", "native_site",
                              "native_record_sha256", "native_files", "link_libraries",
                              "environment", "wheel_filename", "accounting_roots"])
        recipe.update(schema="solarlab.ida75-overlay-recipe.v1", sdk_prefix=str(tmp_path / "absent_sdk"))
        with pytest.raises(ValueError, match="SDK unavailable"):
            builder.validate_recipe(recipe)

    def test_unknown_recipe_fields_are_rejected(self, builder):
        with pytest.raises(ValueError, match="recipe fields"):
            builder.validate_recipe({"schema": "solarlab.ida75-overlay-recipe.v1", "ignore_missing_sdk": True})

    def test_unlisted_source_is_rejected(self, builder, tmp_path):
        (tmp_path / "allowed.h").write_bytes(b"declared")
        expected = {"allowed.h": builder.digest(tmp_path / "allowed.h")}
        builder.verify_files(tmp_path, expected, complete=True)
        (tmp_path / "unexpected.h").write_bytes(b"added after source freeze")
        with pytest.raises(ValueError, match="inventory"):
            builder.verify_files(tmp_path, expected, complete=True)

    def test_runtime_record_checks_bytes_not_just_metadata(self, builder, tmp_path):
        import base64
        payload = tmp_path / "runtime.dylib"
        payload.write_bytes(b"qualified")
        encoded = base64.urlsafe_b64encode(hashlib.sha256(payload.read_bytes()).digest()).decode().rstrip("=")
        record = tmp_path / "RECORD"
        record.write_text(f"runtime.dylib,sha256={encoded},9\nRECORD,,\n")
        pin = builder.digest(record)
        assert builder.verify_record(tmp_path, record, pin) == {"runtime.dylib": builder.digest(payload)}
        payload.write_bytes(b"different")
        with pytest.raises(ValueError, match="payload differs"):
            builder.verify_record(tmp_path, record, pin)

    def test_one_accounting_envelope_includes_tools_and_evidence(self, builder, tmp_path):
        evidence, tools = tmp_path / "evidence", tmp_path / "tools"
        evidence.mkdir()
        tools.mkdir()
        (evidence / "candidate.conda").write_bytes(b"candidate")
        (tools / "compiler").write_bytes(b"compiler")
        assert builder.combined_bytes([str(evidence), str(tools)]) == 17
        with pytest.raises(ValueError, match="disjoint"):
            builder.combined_bytes([str(tmp_path), str(tools)])

    def test_cleanup_failure_retains_owned_process_and_both_errors(self, builder, monkeypatch):
        class Child:
            pid = 43210
            def poll(self):
                return None
            def wait(self, timeout):
                raise builder.subprocess.TimeoutExpired("owned child", timeout)
        def denied(group, sig):
            assert group == Child.pid
            raise PermissionError("signal refused")
        monkeypatch.setattr(builder.os, "killpg", denied)
        result = builder.cleanup_owned(Child())
        assert result["status"] == "cleanup-unverified"
        assert result["pid"] == result["pgid"] == Child.pid
        assert [(e["operation"], e["type"]) for e in result["errors"]] == [
            ("killpg", "PermissionError"), ("wait", "TimeoutExpired")]
        assert result["returncode"] is None

    def test_exited_child_is_not_signalled(self, builder, monkeypatch):
        class Child:
            pid = 43210
            def poll(self):
                return 2
        monkeypatch.setattr(builder.os, "killpg", lambda *a: pytest.fail("exited child was signalled"))
        result = builder.cleanup_owned(Child())
        assert result["status"] == "exited" and result["returncode"] == 2

    def test_build_receipt_survives_cleanup_errors(self, builder, tmp_path, monkeypatch):
        recipe = {"accounting_roots": [str(tmp_path)]}
        plan = {"helper_sha256": builder.digest(Path(builder.__file__)),
                "source_pins_sha256": builder.digest(builder.HERE / "SourcePinsV1.json"),
                "recipe_path": "recipe.json", "recipe_sha256": "a" * 64, "recipe": recipe,
                "work": str(tmp_path), "prepared": {"source": str(tmp_path), "source_hashes": {}},
                "commands": [["synthetic-owned-build-child"]], "environment": {}}
        class Child:
            pid = 43210
            polls = 0
            def poll(self):
                self.polls += 1
                if self.polls == 1:
                    raise RuntimeError("first build error")
                return None
            def wait(self, timeout):
                raise builder.subprocess.TimeoutExpired("owned child", timeout)
        def denied(*args):
            raise PermissionError("signal refused")
        monkeypatch.setattr(builder, "load_pinned", lambda p, s: plan if str(p) == "plan.json" else recipe)
        monkeypatch.setattr(builder, "validate_recipe", lambda r: None)
        monkeypatch.setattr(builder, "verify_files", lambda *a, **k: None)
        monkeypatch.setattr(builder.subprocess, "Popen", lambda *a, **k: Child())
        monkeypatch.setattr(builder.os, "killpg", denied)
        result = builder.run_build(Path("plan.json"), "b" * 64, "msg_1234")
        assert result["failure"] == {"type": "RuntimeError", "message": "first build error"}
        assert result["status"] == "failed_cleanup_unverified"
        assert result["cleanup"]["pgid"] == Child.pid
        assert json.loads((tmp_path / "BuildReceipt.json").read_text()) == result


@pytest.fixture
def native():
    if os.environ.get("SOLARLAB_IDA_OBSERVATION_NATIVE_TEST") != "1":
        pytest.skip("native observation tests require separate Root admission")
    import numpy as np
    from sksundae.ida import IDA
    assert hasattr(IDA, "last_step_snapshot"), "the admitted patched wheel is required"

    def make(callback=None):
        def residual(t, y, yp, out):
            if callback is not None:
                callback()
            out[:] = [yp[0] - (1 + 2*t), y[1] - 2*y[0]]

        def jacobian(t, y, yp, out, cj, jac):
            jac[:] = [[cj, 0], [-2, 1]]

        return IDA(residual, jacfn=jacobian, algebraic_idx=[1],
                   rtol=1e-8, atol=1e-10, first_step=2**-10, max_step=2**-5, max_order=3)
    return np, make


class TestNativeObservation:
    def test_immutable_buffers_clock_authority_and_unchanged_calls(self, native):
        np, make = native
        observed, control = make(), make()
        previous = observed.init_step(0., [1., 2.], [1., 2.])
        control.init_step(0., [1., 2.], [1., 2.])
        packets = []
        for _ in range(64):
            result = observed.step(.125, method="onestep", tstop=.125)
            reference = control.step(.125, method="onestep", tstop=.125)
            assert result.status == reference.status and result.t == reference.t
            assert result.nfev == reference.nfev and result.njev == reference.njev
            assert np.array_equal(result.y, reference.y) and np.array_equal(result.yp, reference.yp)
            packet = observed.last_step_snapshot()
            assert packet["native_before"] == packet["native_after"]
            assert packet["predecessor"]["output_t"] == previous.t
            assert packet["predecessor"]["raw_y"] == previous.y.tobytes()
            assert packet["raw_y"] == result.y.tobytes() and packet["raw_yp"] == result.yp.tobytes()
            assert len(packet["dky"]) == packet["native_before"]["kused"] + 1
            basis = packet["basis"]
            assert basis["kind"] == "ida75_phi_psi_copy_v1" and basis["status"] == 0
            assert basis["build_identity"] == packet["binding"]["identity"]
            assert basis["source_header_sha256"] == "04988c332e2d004b4e838ffca847990f353f68d41cb225f1b9085b3a8bed52ce"
            assert basis["phi_shape"] == (packet["native_before"]["kused"] + 1, 2)
            assert basis["psi_shape"] == (packet["native_before"]["kused"],)
            phi = np.frombuffer(basis["phi"], dtype=basis["dtype"]).reshape(basis["phi_shape"])
            psi = np.frombuffer(basis["psi"], dtype=basis["dtype"])
            assert np.all(np.isfinite(phi)) and np.all(np.isfinite(psi)) and np.all(psi != 0)
            assert phi[0].tobytes() == packet["dky"][0]
            assert float(psi[0]).hex() == packet["native_before"]["hused"].hex()
            for immutable in (phi, psi):
                with pytest.raises(ValueError):
                    immutable.flags.writeable = True
            predecessor = packet["predecessor_dky"]
            assert predecessor["query_t_hex"] == float(previous.t).hex()
            assert predecessor["orders"] == (0, 1)
            assert predecessor["step_before"] == packet["native_before"] == predecessor["step_after"]
            assert predecessor["native_eligible"] == all(s == 0 for s in predecessor["statuses"])
            assert all((status == 0 and type(data) is bytes) or (status != 0 and data is None)
                       for data, status in zip(predecessor["buffers"], predecessor["statuses"]))
            assert all(type(b) is bytes for b in packet["dky"])
            for buffer in packet["dky"] + (packet["raw_y"], packet["raw_yp"]):
                view = np.frombuffer(buffer, dtype=packet["dtype"])
                assert view.shape == packet["shape"] and np.all(np.isfinite(view))
                with pytest.raises(ValueError):
                    view.flags.writeable = True
            with pytest.raises(TypeError):
                packet["native_before"]["tn"] = 0
            exact_gap = sum(Fraction.from_float(x) for x in packet["clock_gap_terms"])
            info = packet["native_before"]
            assert exact_gap == (Fraction.from_float(info["tn"]) - Fraction.from_float(info["hused"])
                                 - Fraction.from_float(previous.t))
            assert observed.last_step_snapshot(expected_step=packet["step_key"])["dky"] == packet["dky"]
            packets.append((packet, packet["raw_y"], packet["dky"]))
            previous = result
            if result.status == 1:
                break
        else:
            pytest.fail("bounded analytic marker was not reached")
        assert any(p[0]["native_before"]["kused"] >= 2 for p in packets)
        assert all(p[0]["raw_y"] == p[1] and p[0]["dky"] == p[2] for p in packets)

    def test_stale_owner_generation_step_and_illegal_query(self, native):
        _, make = native
        solver, other = make(), make()
        with pytest.raises(RuntimeError):
            solver.last_step_snapshot()
        for item in (solver, other):
            item.init_step(0., [1., 2.], [1., 2.])
            item.step(.1, method="onestep")
        key = solver.last_step_snapshot()["step_key"]
        with pytest.raises(RuntimeError):
            other.last_step_snapshot(expected_step=key)
        solver.step(.1, method="onestep")
        with pytest.raises(RuntimeError):
            solver.last_step_snapshot(expected_step=key)
        solver.init_step(0., [1., 2.], [1., 2.])
        solver.step(.1, method="onestep")
        with pytest.raises(RuntimeError):
            solver.last_step_snapshot(expected_step=key)
        with pytest.raises(RuntimeError):
            solver.last_step_snapshot(expected_step=list(key))
        with pytest.raises(TypeError):
            solver.last_step_snapshot(t=float("nan"))

    def test_normal_hidden_predecessor_and_real_recovery(self, native):
        _, make = native
        solver = make()
        solver.init_step(0., [1., 2.], [1., 2.])
        solver.step(.1, method="normal")
        with pytest.raises(RuntimeError) as error:
            solver.last_step_snapshot()
        assert error.value.code in {"predecessor_not_observed", "output_not_at_accepted_time"}
        outputs = []
        for _ in range(3):
            outputs.append(solver.step(.2, method="onestep"))
            try:
                packet = solver.last_step_snapshot()
            except RuntimeError:
                continue
            assert len(outputs) >= 2
            assert packet["predecessor"]["raw_y"] == outputs[-2].y.tobytes()
            break
        else:
            pytest.fail("no eligible consecutive observed endpoints")

    def test_reentrant_observation_is_rejected(self, native):
        _, make = native
        owner, rejected = [], []
        def callback():
            if not rejected:
                with pytest.raises(RuntimeError) as error:
                    owner[0].last_step_snapshot()
                rejected.append(error.value.code)
        owner.append(make(callback))
        owner[0].init_step(0., [1., 2.], [1., 2.])
        owner[0].step(.1, method="onestep")
        assert rejected == ["operation_in_progress"]
