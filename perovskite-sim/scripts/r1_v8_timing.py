"""Coarse exclusive timing with the original trajectory work and boundaries."""
import ast
from contextlib import contextmanager
import inspect
import time

class ExclusiveTimer:
    def __init__(self):
        self.stack = []
        self.entries = {}

    @contextmanager
    def phase(self, name):
        frame = {"name": name, "started": time.monotonic(), "child": 0.0}
        self.stack.append(frame)
        try:
            yield
        finally:
            elapsed = time.monotonic() - frame["started"]
            assert self.stack.pop() is frame
            entry = self.entries.setdefault(name, {"calls": 0, "exclusive_s": 0.0, "inclusive_s": 0.0})
            entry["calls"] += 1
            entry["exclusive_s"] += elapsed - frame["child"]
            entry["inclusive_s"] += elapsed
            if self.stack:
                self.stack[-1]["child"] += elapsed

    def call(self, name, function):
        with self.phase(name):
            return function()

    def report(self, elapsed):
        if self.stack:
            raise ValueError("Unclosed timing scope")
        total = sum(v["exclusive_s"] for v in self.entries.values())
        if not 0 <= total <= elapsed:
            raise ValueError("Exclusive phases do not fit original timing boundary")
        return {"schema": "R1V8ExclusiveTimingV1", "elapsed_s": elapsed,
                "exclusive_phases": self.entries, "exclusive_sum_s": total,
                "remaining_s": elapsed - total,
                "scope": "original_preparation_through_replay_boundary_no_work_removed"}

    def decorate(self, name):
        def decorator(function):
            def wrapped(*args, **kwargs):
                with self.phase(name):
                    return function(*args, **kwargs)
            return wrapped
        return decorator


def wrap_expression(name, node):
    arguments = ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[])
    return ast.copy_location(ast.Call(
        func=ast.Attribute(value=ast.Name(id="_v8_phase_timer", ctx=ast.Load()), attr="call", ctx=ast.Load()),
        args=[ast.Constant(value=name), ast.Lambda(args=arguments, body=node)], keywords=[]), node)


class CoarseInstrumentation(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        node = self.generic_visit(node)
        if node.name == "observe":
            node.decorator_list.append(ast.Call(
                func=ast.Attribute(value=ast.Name(id="_v8_phase_timer", ctx=ast.Load()), attr="decorate", ctx=ast.Load()),
                args=[ast.Constant(value="observer_io_and_bookkeeping")], keywords=[]))
        return node

    def visit_Call(self, node):
        node = self.generic_visit(node)
        target = node.func
        label = None
        if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) and target.value.id == "api":
            key = target.slice.value if isinstance(target.slice, ast.Constant) else None
            label = {"prepare": "prepare_compute", "run": "integrate_compute_and_checks",
                     "replay": "replay_compute", "json_data": "data_conversion",
                     "failure_witness": "failure_witness"}.get(key)
        elif isinstance(target, ast.Name):
            label = {"source_guard": "source_guard", "write": "artifact_write",
                     "canonical": "observer_serialization", "extent": "final_validation",
                     "certificate_check": "final_validation", "precision_record_check": "final_validation"}.get(target.id)
            if target.id == "write" and node.args:
                destination = node.args[0]
                if isinstance(destination, ast.BinOp) and isinstance(destination.right, ast.Constant):
                    label = {"PreparedV1.json": "preparation_write", "ResultV1.json": "result_write",
                             "PhysicsReplayV1.json": "replay_write"}.get(destination.right.value, label)
        return wrap_expression(label, node) if label else node

    def visit_Assign(self, node):
        node = self.generic_visit(node)
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ("persisted", "predicates"):
                node.value = wrap_expression("readback" if name == "persisted" else "four_predicates", node.value)
        return node



def instrument(function, timer):
    """Wrap coarse calls, retaining the supplied function's operations and scope."""
    tree = CoarseInstrumentation().visit(ast.parse(inspect.getsource(function)))
    ast.fix_missing_locations(tree)
    namespace = dict(function.__globals__)
    namespace["_v8_phase_timer"] = timer
    exec(compile(tree, function.__code__.co_filename, "exec"), namespace)
    return namespace[function.__name__]
