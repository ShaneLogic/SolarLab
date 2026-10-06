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
        wrapper = ast.parse((source / "src/sksundae/ida/_solver.py").read_text())
        public = next(node for node in wrapper.body if isinstance(node, ast.ClassDef) and node.name == "IDA")
        assert {"statistics", "last_step_snapshot"} <= {node.name for node in public.body if isinstance(node, ast.FunctionDef)}
        for name, identity in pins["native_cleanup"]["files"].items():
            assert builder.digest(source / "src/sksundae" / name) == identity
        assert "self.LS = sl_SUNLinSol_SuperLUMT(" in code
        assert "self.LS = SUNLinSol_SuperLUMT(" not in code
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

    def test_external_group_cleanup_reaps_direct_child_without_killing_builder(self, builder, monkeypatch):
        called = []
        class Child:
            pid = 43210
            def poll(self):
                return None
            def kill(self):
                called.append("kill_direct_child")
            def wait(self, timeout):
                called.append("wait_direct_child")
                return -9
        monkeypatch.setattr(builder.os, "getpgrp", lambda: 12345)
        monkeypatch.setattr(builder.os, "killpg", lambda *a: pytest.fail("would kill the builder before its receipt"))
        result = builder.cleanup_owned(Child(), external_process_group=True)
        assert called == ["kill_direct_child", "wait_direct_child"]
        assert result["pgid"] == 12345 and result["direct_child_reaped"]
        assert result["group_cleanup_owner"] == "external_supervisor" and result["group_empty"] is None

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

    def test_overlay_materializes_new_extension_but_reuses_original_payload(self, builder, tmp_path, monkeypatch):
        import zipfile
        work, site = tmp_path / "work", tmp_path / "original"
        work.mkdir()
        library = site / "sksundae/.dylibs/qualified.dylib"
        library.parent.mkdir(parents=True)
        library.write_bytes(b"qualified-library-metadata-fixture")
        meta = site / "scikit_sundae-1.1.3.dist-info/WHEEL"
        meta.parent.mkdir()
        meta.write_bytes(b"original-wheel-metadata")
        source = work / "source"
        header = source / "src/sksundae/ida_observation_copy.h"
        header.parent.mkdir(parents=True)
        header.write_bytes(b"source-metadata-fixture")
        extension = work / "_cy_ida.cpython-313-darwin.so"
        extension.write_bytes(b"compiled-image-metadata-fixture")
        monkeypatch.setattr(builder, "pins", lambda: {"files": {}, "native_cleanup": {"files": {}}})
        result = builder.pack_overlay({"work": str(work), "extension": str(extension),
            "prepared": {"source": str(source)}, "recipe": {"native_site": str(site),
            "native_files": {str(p.relative_to(site)): builder.digest(p) for p in (library, meta)},
            "wheel_filename": "metadata-fixture.whl"}})
        overlay = Path(result["overlay"])
        image = overlay / "sksundae/_cy_ida.cpython-313-darwin.so"
        assert image.is_file() and not image.is_symlink()
        assert image.read_bytes() == extension.read_bytes()
        borrowed = overlay / library.relative_to(site)
        assert borrowed.is_symlink() and borrowed.resolve() == library.resolve()
        assert borrowed.read_bytes() == b"qualified-library-metadata-fixture"
        with zipfile.ZipFile(result["wheel"]) as archive:
            assert archive.read("sksundae/_cy_ida.cpython-313-darwin.so") == image.read_bytes()
        record = overlay / "scikit_sundae-1.1.3.dist-info/RECORD"
        builder.verify_record(overlay, record, builder.digest(record))
        installed = builder.install_verified_wheel(result, work / "installed")
        destination = Path(installed["installed"])
        assert installed["installed_files"] == result["wheel_files"]
        assert not any(p.is_symlink() for p in destination.rglob("*"))
        assert (destination / library.relative_to(site)).read_bytes() == library.read_bytes()
        with pytest.raises(ValueError, match="installation exists"):
            builder.install_verified_wheel(result, destination)


@pytest.fixture
def native():
    if os.environ.get("SOLARLAB_IDA_OBSERVATION_NATIVE_TEST") != "1":
        pytest.skip("native observation tests require separate Root admission")
    import numpy as np
    from sksundae.ida import IDA
    assert hasattr(IDA, "last_step_snapshot"), "the admitted patched wheel is required"
    assert callable(getattr(IDA, "statistics", None)), "both reviewed APIs must exist before construction"

    def make(callback=None, *, sparse=False, cj_log=None, **options):
        def residual(t, y, yp, out):
            if callback is not None:
                callback()
            out[:] = [yp[0] - (1 + 2*t), y[1] - 2*y[0]]

        def jacobian(t, y, yp, out, cj, jac):
            if cj_log is not None:
                cj_log.append((float(t), float(cj)))
            jac[:] = [cj, -2, 1] if sparse else [[cj, 0], [-2, 1]]

        settings = dict(jacfn=jacobian, algebraic_idx=[1], rtol=1e-8, atol=1e-10,
                        first_step=2**-10, max_step=2**-5, max_order=3)
        if sparse:
            settings.update(linsolver="sparse", sparsity=np.array([[1, 0], [1, 1]]), nthreads=1)
        settings.update(options)
        return IDA(residual, **settings)
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


class TestNativeCompositionLifetime:
    """Binding-only sparse lifetime cases, never a SolarLab device protocol."""

    def test_sparse_init_statistics_and_normal_no_factorization_destruction(self, native):
        import gc
        import weakref
        np, make = native
        called = []
        solver = make(lambda: called.append("residual"), sparse=True)
        y, yp = np.array([1., 2.]), np.array([1., 2.])
        y.flags.writeable = yp.flags.writeable = False
        original = (y.tobytes(), yp.tobytes())
        result = solver.init_step(0., y, yp)
        assert result.success and result.status == 0
        stats = solver.statistics()
        assert stats["num_steps"] == stats["linear_setups"] == stats["jacobian_evals"] == 0
        assert stats["method_fields_valid"] is False
        assert stats["nonlin_conv_coef_requested"] is None
        assert stats["coefficient_getter_available"] is False
        assert stats["observation_generation"] == 1
        with pytest.raises(RuntimeError, match=r"^no_accepted_interval:"):
            solver.last_step_snapshot()
        assert stats == solver.statistics()
        assert not called and original == (y.tobytes(), yp.tobytes())
        reference = weakref.ref(solver)
        del solver
        gc.collect()
        assert reference() is None
        print("NO_FACTORIZATION_NORMAL_DESTRUCTION", json.dumps(stats, allow_nan=False))

    def test_failed_atol_setup_preserves_exception_and_cleans_partial_solver(self, native):
        import gc
        import weakref
        _, make = native
        solver = make(sparse=True, atol=[1e-10])
        # _set_tolerances is reached after the real sparse LS and IDA allocations.
        with pytest.raises(ValueError, match="'atol' length .* differs from problem size") as error:
            solver.init_step(0., [1., 2.], [1., 2.])
        assert "(1)" in str(error.value) and "(2)" in str(error.value)
        with pytest.raises(RuntimeError, match="must be initialized"):
            solver.statistics()
        with pytest.raises(RuntimeError):
            solver.last_step_snapshot()
        # The caught traceback can hold solver locals; release it before deallocation.
        del error
        reference = weakref.ref(solver)
        del solver
        gc.collect()
        assert reference() is None
        print("PARTIAL_ATOL_FAILURE_NORMAL_DESTRUCTION")

    def test_sparse_factorization_statistics_snapshot_and_default_none_agree(self, native):
        import gc
        import weakref
        np, make = native
        callback_cj = []
        default = make(sparse=True, cj_log=callback_cj)
        explicit_none = make(sparse=True, nonlin_conv_coef=None)
        results, stats, packets = [], [], []
        for solver in (default, explicit_none):
            solver.init_step(0., [1., 2.], [1., 2.])
            result = solver.step(.125, method="onestep", tstop=.125)
            assert result.success
            snapshot = solver.last_step_snapshot()
            statistics = solver.statistics()
            assert statistics == solver.statistics()
            assert solver.last_step_snapshot(expected_step=snapshot["step_key"])["basis"] == snapshot["basis"]
            assert statistics["num_steps"] > 0 and statistics["linear_setups"] > 0
            assert statistics["jacobian_evals"] > 0 and statistics["method_fields_valid"]
            assert statistics["observation_owner"] == snapshot["owner"]
            assert statistics["observation_generation"] == snapshot["generation"]
            results.append(result)
            stats.append(statistics)
            packets.append(snapshot)
        assert callback_cj and all(np.isfinite(cj) and cj > 0 for _, cj in callback_cj)
        assert results[0].t == results[1].t and results[0].status == results[1].status
        assert results[0].y.tobytes() == results[1].y.tobytes()
        assert results[0].yp.tobytes() == results[1].yp.tobytes()
        assert {k: v for k, v in stats[0].items() if k != "observation_owner"} == {
            k: v for k, v in stats[1].items() if k != "observation_owner"}
        old_bytes = packets[0]["raw_y"]
        refs = [weakref.ref(default), weakref.ref(explicit_none)]
        del solver, default, explicit_none
        gc.collect()
        assert all(ref() is None for ref in refs)
        assert packets[0]["raw_y"] == old_bytes
        print("FACTORIZATION_NORMAL_DESTRUCTION", json.dumps({
            "statistics": stats, "callback_cj": callback_cj,
            "binding": dict(packets[0]["binding"])}, allow_nan=False))

    @pytest.mark.parametrize("coefficient", [True, False, "0.2", 0., -1., float("nan"), float("inf"), -float("inf")])
    def test_invalid_coefficient_is_rejected_before_initialization(self, native, coefficient):
        _, make = native
        with pytest.raises((TypeError, ValueError), match="nonlin_conv_coef"):
            make(sparse=True, nonlin_conv_coef=coefficient)

    def test_explicit_coefficient_and_reentrant_statistics_guard(self, native):
        _, make = native
        owner, rejected = [], []
        def callback():
            if not rejected:
                with pytest.raises(RuntimeError) as error:
                    owner[0].statistics()
                rejected.append(error.value.code)
        owner.append(make(callback, sparse=True, nonlin_conv_coef=0.2))
        owner[0].init_step(0., [1., 2.], [1., 2.])
        assert owner[0].statistics()["nonlin_conv_coef_requested"] == 0.2
        owner[0].step(.125, method="onestep", tstop=.125)
        assert rejected == ["operation_in_progress"]
        assert owner[0].statistics()["coefficient_getter_available"] is False
        owner.clear()

    def test_reinit_and_batch_statistics_windows_remain_explicit(self, native):
        import gc
        import weakref
        _, make = native
        solver = make(sparse=True)
        solver.init_step(0., [1., 2.], [1., 2.])
        solver.step(.125, method="onestep", tstop=.125)
        packet = solver.last_step_snapshot()
        original_history = (packet["raw_y"], packet["basis"]["phi"], packet["predecessor"]["raw_y"])
        solver.init_step(0., [1., 2.], [1., 2.])
        stats = solver.statistics()
        assert stats["num_steps"] == 0 and stats["method_fields_valid"] is False
        assert stats["observation_owner"] == packet["owner"]
        assert stats["observation_generation"] == packet["generation"] + 1
        with pytest.raises(RuntimeError):
            solver.last_step_snapshot(expected_step=packet["step_key"])
        # IDAReInit retains the LS: its next initialize must retire old factors
        # before the native refact=NO path creates replacements.
        second = solver.step(.125, method="onestep", tstop=.125)
        assert second.success
        after = solver.statistics()
        second_packet = solver.last_step_snapshot()
        assert after["num_steps"] > 0 and after["linear_setups"] > 0 and after["jacobian_evals"] > 0
        assert second_packet["owner"] == packet["owner"]
        assert second_packet["generation"] == packet["generation"] + 1
        assert second_packet["step_key"] != packet["step_key"]
        reference = weakref.ref(solver)
        del solver
        gc.collect()
        assert reference() is None
        assert (packet["raw_y"], packet["basis"]["phi"], packet["predecessor"]["raw_y"]) == original_history
        print("REINIT_SECOND_FACTORIZATION_NORMAL_DESTRUCTION", json.dumps({"before_step": stats, "after_step": after}, allow_nan=False))
        # Use a fresh dense marker for the existing batch-solve access limitation.
        batch = make()
        assert batch.solve([0., 2**-10], [1., 2.], [1., 2.]).success
        with pytest.raises(RuntimeError, match="must be initialized"):
            batch.statistics()
