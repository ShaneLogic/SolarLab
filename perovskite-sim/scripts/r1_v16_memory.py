"""Optional, bounded observations of live runner roots at explicit phase edges.

This module is diagnostic only. It neither serializes numerical values nor
estimates a full object graph. Current RSS and the process high-water mark are
different measurements; external sampling is needed to see between edges.
"""
from __future__ import annotations

from dataclasses import fields, is_dataclass
import ctypes
import itertools
import json
import os
from pathlib import Path
import resource
import sys
import time
from types import GetSetDescriptorType, MemberDescriptorType

import numpy as np


SCHEMA = "R1V16NativeMemoryObservationV1"
ROOT_NAMES = frozenset(("raw_result", "result", "rows", "persisted", "prepared", "replay"))
LIMITS = {"max_nodes_per_root": 256, "max_children_per_node": 16, "max_depth": 4}
_PRIORITY_KEYS = ("accepted_steps", "output_states", "state", "physics_reconstruction")


class MemoryObservationError(RuntimeError):
    """A requested memory diagnostic failed and cannot qualify the run."""


class _ProcTaskInfo(ctypes.Structure):
    # sys/proc_info.h: struct proc_taskinfo, PROC_PIDTASKINFO = 4.
    _fields_ = [(name, ctypes.c_uint64) for name in (
        "virtual_size", "resident_size", "total_user", "total_system",
        "threads_user", "threads_system",
    )] + [(name, ctypes.c_int32) for name in (
        "policy", "faults", "pageins", "cow_faults", "messages_sent",
        "messages_received", "syscalls_mach", "syscalls_unix", "csw",
        "threadnum", "numrunning", "priority",
    )]


def _current_rss_reader():
    """Construct one cheap current-RSS reader; unsupported platforms fail."""
    if sys.platform == "darwin":
        library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        query = library.proc_pidinfo
        query.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
                          ctypes.c_void_p, ctypes.c_int]
        query.restype = ctypes.c_int

        def current():
            value = _ProcTaskInfo()
            expected = ctypes.sizeof(value)
            received = query(os.getpid(), 4, 0, ctypes.byref(value), expected)
            if received != expected:
                raise OSError(ctypes.get_errno(), "proc_pidinfo returned an incomplete task record")
            return int(value.resident_size)

        return current, "proc_pidinfo_PROC_PIDTASKINFO.pti_resident_size_bytes"
    if sys.platform.startswith("linux"):
        page_size = os.sysconf("SC_PAGE_SIZE")

        def current():
            # statm is a bounded kernel record, not a heap walk.
            with open("/proc/self/statm", encoding="ascii") as stream:
                words = stream.read(4096).split()
            if len(words) < 2:
                raise ValueError("incomplete /proc/self/statm record")
            return int(words[1]) * page_size

        return current, "proc_self_statm_resident_pages_times_SC_PAGE_SIZE"
    raise ValueError("current RSS is unsupported on this platform")


def _high_water_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _type_name(value):
    cls = type(value)
    return cls.__module__ + "." + cls.__qualname__


def _stored_dataclass_fields(value):
    """Read stored dataclass fields only, without calling properties/to_dict."""
    if isinstance(value, type) or not is_dataclass(value):
        return None
    declared = fields(value)
    result = []
    namespace_descriptor = next((base.__dict__["__dict__"] for base in type(value).__mro__
                                 if "__dict__" in base.__dict__), None)
    namespace = (namespace_descriptor.__get__(value, type(value))
                 if isinstance(namespace_descriptor, GetSetDescriptorType) else {})
    for field in itertools.islice(declared, LIMITS["max_children_per_node"]):
        if field.name in namespace:
            result.append((field.name, namespace[field.name]))
            continue
        descriptor = next((base.__dict__.get(field.name) for base in type(value).__mro__
                           if field.name in base.__dict__), None)
        if isinstance(descriptor, MemberDescriptorType):
            result.append((field.name, descriptor.__get__(value, type(value))))
    return len(declared), result


def _sequence_indices(length, limit):
    if length <= limit:
        return range(length)
    # Observe endpoints and evenly spaced rows without copying the full list.
    return tuple(i * (length - 1) // (limit - 1) for i in range(limit))


def _children(value):
    """Return no more than the fixed child limit, plus an exact direct count."""
    limit = LIMITS["max_children_per_node"]
    if type(value) is dict:
        result = [(key, value[key]) for key in _PRIORITY_KEYS if key in value]
        chosen = {key for key, _ in result}
        for key, child in value.items():
            if len(result) >= limit:
                break
            if key not in chosen:
                result.append((key, child))
        return len(value), result
    if type(value) in (list, tuple):
        return len(value), [(index, value[index]) for index in _sequence_indices(len(value), limit)]
    stored = _stored_dataclass_fields(value)
    return stored if stored is not None else (0, [])


def _descriptor(value):
    """Bounded metadata only; no array data or scientific scalar values."""
    result = {"type": _type_name(value), "present": value is not None}
    if type(value) in (dict, list, tuple, str, bytes, bytearray):
        result["length"] = len(value)
        result["shallow_bytes"] = sys.getsizeof(value)
    elif isinstance(value, np.ndarray):
        result.update(dtype=str(value.dtype), shape=list(value.shape),
                      logical_nbytes=int(value.nbytes), owns_data=bool(value.flags.owndata))
    elif isinstance(value, np.generic):
        result.update(dtype=str(value.dtype), scalar_nbytes=int(value.dtype.itemsize))
    elif type(value) in (type(None), bool, int, float, complex):
        result["shallow_bytes"] = sys.getsizeof(value)
    return result


def summarize_root(value):
    """Inspect a fixed sample, with at most 256 identities retained temporarily.

    OWNDATA bytes are exact only for sampled distinct arrays within this root.
    They are neither resident bytes nor additive between roots. Views report
    logical bytes separately; their base allocations are not followed.
    """
    descriptor = _descriptor(value)
    direct_count, direct = _children(value)
    descriptor["direct_child_count"] = direct_count
    descriptor["direct_children_sample"] = [
        {"key": str(key)[:160] if type(key) in (str, int) else "<" + _type_name(key) + ">",
         **_descriptor(child)} for key, child in direct
    ]
    counts = {"dict": 0, "list": 0, "tuple": 0, "ndarray": 0,
              "numpy_scalar": 0, "python_float": 0, "other": 0}
    sample = {"unique_nodes": 0, "duplicate_edges": 0, "unvisited_children": 0,
              "depth_limited_nodes": 0, "node_limit_reached": False,
              "sampled_owning_ndarray_nbytes": 0, "sampled_view_logical_nbytes": 0,
              "sampled_numpy_scalar_nbytes": 0, "dtype_counts": {}, "type_counts": counts}
    seen = set()
    pending = [(value, 0)]
    while pending:
        child, depth = pending.pop()
        identity = id(child)
        if identity in seen:
            sample["duplicate_edges"] += 1
            continue
        if len(seen) >= LIMITS["max_nodes_per_root"]:
            sample["node_limit_reached"] = True
            break
        seen.add(identity)
        if isinstance(child, np.ndarray):
            counts["ndarray"] += 1
            key = "sampled_owning_ndarray_nbytes" if child.flags.owndata else "sampled_view_logical_nbytes"
            sample[key] += int(child.nbytes)
            dtype = str(child.dtype)
            sample["dtype_counts"][dtype] = sample["dtype_counts"].get(dtype, 0) + 1
        elif isinstance(child, np.generic):
            counts["numpy_scalar"] += 1
            sample["sampled_numpy_scalar_nbytes"] += int(child.dtype.itemsize)
        else:
            label = {dict: "dict", list: "list", tuple: "tuple", float: "python_float"}.get(type(child), "other")
            counts[label] += 1
            total, selected = _children(child)
            if depth >= LIMITS["max_depth"]:
                sample["depth_limited_nodes"] += bool(total)
                sample["unvisited_children"] += total
            else:
                sample["unvisited_children"] += total - len(selected)
                pending.extend((item, depth + 1) for _, item in reversed(selected))
    sample["unique_nodes"] = len(seen)
    descriptor["bounded_sample"] = sample
    descriptor["sample_scope"] = "bounded_identity_deduplicated_sample_not_total_graph_or_RSS_estimate"
    return descriptor


class NativeMemoryObserver:
    """Callable ``observer(phase, begin|end|error, roots)`` writing durable JSONL.

    Construction is explicit and creates a new file. No live root, traceback or
    exception is saved on this object. A failed collector is sticky and raises;
    the caller must mark its diagnostic gate failed even if it preserves the
    ordinary run artifacts. ``close`` fails for an unclosed phase.
    """
    def __init__(self, path):
        self.path = Path(path)
        self._stack = []
        self._sequence = 0
        self._bytes = 0
        self._elapsed_s = 0.0
        self._failure = None
        self._closed = False
        self._rss, self._rss_source = _current_rss_reader()
        self._stream = self.path.open("x", encoding="utf-8", newline="\n")

    def __call__(self, phase, event, roots):
        started = time.monotonic()
        try:
            if self._closed or self._failure is not None:
                raise ValueError("memory observer is closed or has already failed")
            if type(phase) is not str or not phase or len(phase) > 160:
                raise ValueError("memory phase must be a bounded nonempty string")
            if event not in ("begin", "end", "error"):
                raise ValueError("memory event must be begin, end or error")
            if type(roots) is not dict or not set(roots).issubset(ROOT_NAMES):
                raise ValueError("memory roots must use the declared runner root names")
            if event != "begin" and (not self._stack or self._stack[-1] != phase):
                raise ValueError("memory phase boundaries are unbalanced")
            current = self._rss()
            high_water = _high_water_bytes()
            if type(current) is not int or current <= 0 or high_water <= 0:
                raise ValueError("invalid native RSS measurement")
            names = list(roots)
            record = {"schema": SCHEMA, "sequence": self._sequence,
                "pid": os.getpid(), "monotonic_s": started, "phase": phase, "event": event,
                "current_rss_bytes": current, "cumulative_high_water_rss_bytes": high_water,
                "rss_source": self._rss_source, "limits": LIMITS,
                "roots": {name: summarize_root(value) for name, value in roots.items()},
                "same_nonnull_root_identity": [[a, b] for index, a in enumerate(names)
                    for b in names[index + 1:] if roots[a] is not None and roots[a] is roots[b]],
                "scope": "phase_edge_RSS_with_bounded_live_roots_no_forced_GC_no_physics_evaluation",
                "peak_scope": "current_is_phase_edge_only_high_water_is_cumulative_never_subtract"}
            raw = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
            self._stream.write(raw)
            self._stream.flush()
            os.fsync(self._stream.fileno())
            self._bytes += len(raw.encode("utf-8"))
            self._sequence += 1
            if event == "begin":
                self._stack.append(phase)
            else:
                self._stack.pop()
        except Exception as exc:
            if self._failure is None:
                self._failure = {"type": type(exc).__name__, "message": str(exc)}
            raise MemoryObservationError("native memory observation failed: " + str(exc)) from exc
        except KeyboardInterrupt:
            self._failure = {"type": "KeyboardInterrupt", "message": "memory observation interrupted"}
            raise
        finally:
            self._elapsed_s += time.monotonic() - started

    def summary(self):
        return {"schema": "R1V16NativeMemoryObserverSummaryV1", "event_count": self._sequence,
                "artifact_bytes": self._bytes, "collector_elapsed_s": self._elapsed_s,
                "closed": self._closed, "open_phases": list(self._stack),
                "failure": None if self._failure is None else dict(self._failure),
                "passed": self._failure is None and not self._stack and self._sequence > 0,
                "limits": dict(LIMITS), "rss_source": self._rss_source,
                "roots_retained": False, "forced_gc": False,
                "scope": "diagnostic_only_no_resource_admission_or_scientific_qualification"}

    def close(self):
        if self._closed:
            return
        try:
            try:
                self._stream.flush()
                os.fsync(self._stream.fileno())
                if self._stack:
                    raise ValueError("memory observer has unclosed phases")
            finally:
                self._stream.close()
        except Exception as exc:
            if self._failure is None:
                self._failure = {"type": type(exc).__name__, "message": str(exc)}
            raise MemoryObservationError("native memory observation close failed: " + str(exc)) from exc
        finally:
            self._closed = True
