/* Source-owned copy boundary for SUNDIALS 7.5.0 IDA, double/int32/serial.
 * The private declaration is the hash-pinned upstream ida_impl.h, never a
 * reconstructed structure. ABI provenance remains a separate review gate.
 * No pointer, arithmetic reconstruction, or mutable solver buffer escapes.
 */
#ifndef SOLARLAB_IDA75_OBSERVATION_COPY_H
#define SOLARLAB_IDA75_OBSERVATION_COPY_H

#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <nvector/nvector_serial.h>
#include <sundials/sundials_version.h>
#include <sunnonlinsol/sunnonlinsol_newton.h>
#include "ida_impl.h"

#if SUNDIALS_VERSION_MAJOR != 7 || SUNDIALS_VERSION_MINOR != 5 || SUNDIALS_VERSION_PATCH != 0
#error "The observation copy requires the pinned SUNDIALS 7.5.0 headers"
#endif
#if !defined(SUNDIALS_DOUBLE_PRECISION) || !defined(SUNDIALS_INT32_T)
#error "The observation copy requires double precision and int32 indices"
#endif
_Static_assert(sizeof(sunrealtype) == 8, "binary64 storage required");
_Static_assert(sizeof(sunindextype) == 4, "int32 storage required");

/* These are boundary statuses, not IDA status values. */
enum {
    SL_OBS_ARGUMENT = -7001, SL_OBS_VERSION = -7002,
    SL_OBS_GETTER = -7003, SL_OBS_STAMP = -7004,
    SL_OBS_VECTOR = -7005, SL_OBS_NONFINITE = -7006,
    SL_OBS_ALIAS = -7007
};

static const char *sl_ida75_library_path(void)
{
    Dl_info info;
    if (!dladdr((const void *)&IDAGetCurrentTime, &info)) return NULL;
    return info.dli_fname;
}

static int sl_ida75_same_word(sunrealtype a, sunrealtype b)
{
    return memcmp(&a, &b, sizeof(a)) == 0;
}

static int sl_ida75_overlap(const void *a, size_t na, const void *b, size_t nb)
{
    uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
    if (na > UINTPTR_MAX - x || nb > UINTPTR_MAX - y) return 1;
    return x < y + nb && y < x + na;
}

/* Check supported public scalars before dereferencing any private vector.
 * Matching these is a mismatch detector, not a general private-ABI proof.
 */
static int sl_ida75_stamp(void *memory, long ns, sunrealtype tn,
                         sunrealtype hu, int qu, sunrealtype hn, int qn)
{
    int major, minor, patch, status, actual_q, actual_next_q;
    long actual_ns;
    sunrealtype actual_t, actual_h, actual_next_h;
    char label[128];
    IDAMem m;
    if (!memory || ns < 1 || qu < 1 || qu >= MXORDP1 || qn < 1 || qn >= MXORDP1
        || !isfinite(tn) || !isfinite(hu) || !isfinite(hn) || hu == 0 || hn == 0)
        return SL_OBS_ARGUMENT;
    status = SUNDIALSGetVersionNumber(&major, &minor, &patch, label, sizeof(label));
    if (status != 0 || major != 7 || minor != 5 || patch != 0) return SL_OBS_VERSION;
    if (IDAGetNumSteps(memory, &actual_ns) != 0
        || IDAGetCurrentTime(memory, &actual_t) != 0
        || IDAGetLastStep(memory, &actual_h) != 0
        || IDAGetLastOrder(memory, &actual_q) != 0
        || IDAGetCurrentStep(memory, &actual_next_h) != 0
        || IDAGetCurrentOrder(memory, &actual_next_q) != 0) return SL_OBS_GETTER;
    if (actual_ns != ns || actual_q != qu || actual_next_q != qn
        || !sl_ida75_same_word(actual_t, tn) || !sl_ida75_same_word(actual_h, hu)
        || !sl_ida75_same_word(actual_next_h, hn)) return SL_OBS_STAMP;
    m = (IDAMem)memory;
    if (m->ida_nst != ns || m->ida_kused != qu || m->ida_kk != qn
        || !sl_ida75_same_word(m->ida_tn, tn) || !sl_ida75_same_word(m->ida_hused, hu)
        || !sl_ida75_same_word(m->ida_hh, hn)
        || !m->ida_MallocDone || m->ida_maxord_alloc < qu
        || !isfinite(m->ida_uround) || m->ida_uround <= 0) return SL_OBS_STAMP;
    return 0;
}

/* Caller owns exact-capacity, disjoint output arrays. All inputs are validated
 * before any copy. Only q+1 phi vectors and q psi words are needed by GetDky.
 */
static int sl_ida75_copy(void *memory, sunindextype n, long ns,
                        sunrealtype tn, sunrealtype hu, int qu,
                        sunrealtype hn, int qn,
                        sunrealtype *phi, size_t phi_count,
                        sunrealtype *psi, size_t psi_count,
                        sunrealtype *uround)
{
    IDAMem m;
    int i, status;
    sunindextype j;
    size_t row_bytes, phi_bytes, psi_bytes;
    sunrealtype *rows[MXORDP1];
    if (!phi || !psi || !uround || n <= 0 || qu < 1 || qu >= MXORDP1)
        return SL_OBS_ARGUMENT;
    if ((size_t)n > SIZE_MAX / sizeof(sunrealtype) / (size_t)(qu + 1))
        return SL_OBS_ARGUMENT;
    if (phi_count != (size_t)n * (size_t)(qu + 1) || psi_count != (size_t)qu)
        return SL_OBS_ARGUMENT;
    row_bytes = (size_t)n * sizeof(sunrealtype);
    phi_bytes = phi_count * sizeof(sunrealtype);
    psi_bytes = psi_count * sizeof(sunrealtype);
    if (sl_ida75_overlap(phi, phi_bytes, psi, psi_bytes)
        || sl_ida75_overlap(phi, phi_bytes, uround, sizeof(*uround))
        || sl_ida75_overlap(psi, psi_bytes, uround, sizeof(*uround))) return SL_OBS_ALIAS;
    status = sl_ida75_stamp(memory, ns, tn, hu, qu, hn, qn);
    if (status) return status;
    m = (IDAMem)memory;
    if (sl_ida75_overlap(phi, phi_bytes, m, sizeof(*m))
        || sl_ida75_overlap(psi, psi_bytes, m, sizeof(*m))
        || sl_ida75_overlap(uround, sizeof(*uround), m, sizeof(*m))) return SL_OBS_ALIAS;
    for (i = 0; i <= qu; ++i) {
        N_Vector vector = m->ida_phi[i];
        if (!vector || N_VGetVectorID(vector) != SUNDIALS_NVEC_SERIAL
            || N_VGetLength_Serial(vector) != n) return SL_OBS_VECTOR;
        rows[i] = N_VGetArrayPointer(vector);
        if (!rows[i]) return SL_OBS_VECTOR;
        if (sl_ida75_overlap(phi, phi_bytes, rows[i], row_bytes)
            || sl_ida75_overlap(psi, psi_bytes, rows[i], row_bytes)
            || sl_ida75_overlap(uround, sizeof(*uround), rows[i], row_bytes)) return SL_OBS_ALIAS;
        for (j = 0; j < n; ++j) if (!isfinite(rows[i][j])) return SL_OBS_NONFINITE;
    }
    for (i = 0; i < qu; ++i) {
        if (!isfinite(m->ida_psi[i]) || m->ida_psi[i] == 0) return SL_OBS_NONFINITE;
    }
    for (i = 0; i <= qu; ++i) memcpy(phi + (size_t)i * n, rows[i], row_bytes);
    memcpy(psi, m->ida_psi, psi_bytes);
    memcpy(uround, &m->ida_uround, sizeof(*uround));
    return sl_ida75_stamp(memory, ns, tn, hu, qu, hn, qn);
}

/* epcon is initialized at creation. The remaining fields are read only after
 * a successful endpoint in this generation and an actual NLS iteration.
 * Reinitialization or a failed return must not expose earlier method state.
 * The caller publishes unavailable fields as None, never as copied sentinels.
 */
static int sl_ida75_nls_state(void *memory, long ns, long nni,
                            sunrealtype tn, sunrealtype hu, int qu,
                            sunrealtype hn, int qn, int read_step,
                            sunrealtype *values, size_t count)
{
    int major, minor, patch, status, i;
    long actual_ns, actual_nni, actual_failures;
    sunrealtype actual_t, fields[5];
    size_t copied = read_step ? 5 : 1;
    char label[128];
    IDAMem m;
    if (!memory || !values || count != 5 || ns < 0 || nni < 0
        || !isfinite(tn) || (read_step != 0 && read_step != 1)
        || (read_step && (ns < 1 || nni < 1))) return SL_OBS_ARGUMENT;
    status = SUNDIALSGetVersionNumber(&major, &minor, &patch, label, sizeof(label));
    if (status != 0 || major != 7 || minor != 5 || patch != 0) return SL_OBS_VERSION;
    if (IDAGetNumSteps(memory, &actual_ns) != 0
        || IDAGetCurrentTime(memory, &actual_t) != 0
        || IDAGetNonlinSolvStats(memory, &actual_nni, &actual_failures) != 0)
        return SL_OBS_GETTER;
    if (actual_ns != ns || actual_nni != nni || !sl_ida75_same_word(actual_t, tn))
        return SL_OBS_STAMP;
    m = (IDAMem)memory;
    if (!m->ida_MallocDone || m->ida_nst != ns || m->ida_nni != nni
        || !sl_ida75_same_word(m->ida_tn, tn)) return SL_OBS_STAMP;
    if (sl_ida75_overlap(values, count * sizeof(*values), m, sizeof(*m)))
        return SL_OBS_ALIAS;
    if (read_step) {
        status = sl_ida75_stamp(memory, ns, tn, hu, qu, hn, qn);
        if (status) return status;
    }
    memcpy(&fields[0], &m->ida_epcon, sizeof(*fields));
    if (read_step) {
        memcpy(&fields[1], &m->ida_epsNewt, sizeof(*fields));
        memcpy(&fields[2], &m->ida_ss, sizeof(*fields));
        memcpy(&fields[3], &m->ida_oldnrm, sizeof(*fields));
        memcpy(&fields[4], &m->ida_toldel, sizeof(*fields));
    }
    for (i = 0; i < (int)copied; ++i)
        if (!isfinite(fields[i]) || fields[i] < 0) return SL_OBS_NONFINITE;
    /* A positive subnormal epcon can legitimately underflow toldel to zero. */
    if (fields[0] <= 0 || (read_step && fields[1] <= 0))
        return SL_OBS_NONFINITE;
    memcpy(values, fields, copied * sizeof(*values));
    if (m->ida_nni != nni) return SL_OBS_STAMP;
    return read_step ? sl_ida75_stamp(memory, ns, tn, hu, qu, hn, qn) : 0;
}
/* The optional guard below owns only its delegate and bounded trace. The IDA
 * memory, Newton solver, state, linear solver and callback data remain owned
 * by the original integrator. Installing it changes a numerical policy; the
 * read-only observation functions above do not install it.
 */
enum {
    SL_GUARD_OWNER = -7010, SL_GUARD_TRACE_FULL = -7011,
    SL_GUARD_VALUE = -7012, SL_GUARD_ALLOCATION = -7013
};
#define SL_GUARD_MAX_RECORDS 4096

typedef struct {
    long generation, operation, sequence, step;
    int iteration, iteration_status, default_status, returned_status, override;
    int oldnrm_before_valid, oldnrm_after_valid, norm_observed, norm_valid;
    sunrealtype norm, tolerance, ss_before, ss_after, oldnrm_before, oldnrm_after;
    sunrealtype toldel, time, cj;
} sl_ida75_guard_record;

typedef struct {
    IDAMem memory;
    SUNNonlinearSolver nls;
    SUNNonlinSolConvTestFn original;
    void *original_data;
    sunindextype size;
    long generation, operation, calls;
    size_t capacity, count;
    int active, overflow, native_status, oldnrm_valid;
    sl_ida75_guard_record *records, first_unrecorded;
} sl_ida75_guard;

static int sl_ida75_guard_callback(SUNNonlinearSolver nls, N_Vector ycor,
                                  N_Vector delta, sunrealtype tol,
                                  N_Vector weights, void *data)
{
    sl_ida75_guard *g = (sl_ida75_guard *)data;
    sl_ida75_guard_record row = {0};
    int valid_vectors;
    if (!g || !g->memory || !g->original || g->nls != nls || !g->active
        || g->original_data != (void *)g->memory) return SL_GUARD_OWNER;
    row.generation = g->generation;
    row.operation = g->operation;
    row.sequence = ++g->calls;
    row.step = g->memory->ida_nst;
    row.iteration = -1;
    row.iteration_status = SUNNonlinSolGetCurIter(nls, &row.iteration);
    row.tolerance = tol;
    row.ss_before = g->memory->ida_ss;
    row.oldnrm_before_valid = g->oldnrm_valid;
    if (g->oldnrm_valid) row.oldnrm_before = g->memory->ida_oldnrm;
    row.toldel = g->memory->ida_toldel;
    row.time = g->memory->ida_tn;
    row.cj = g->memory->ida_cj;

    /* Exactly one invocation of the captured native callback and original
     * data. Its oldnrm/ss updates and non-success statuses are authoritative.
     */
    row.default_status = g->original(nls, ycor, delta, tol, weights,
                                     g->original_data);
    row.returned_status = row.default_status;
    row.ss_after = g->memory->ida_ss;
    if (row.iteration_status == SUN_SUCCESS && row.iteration == 0)
        g->oldnrm_valid = 1;
    row.oldnrm_after_valid = g->oldnrm_valid;
    if (g->oldnrm_valid) row.oldnrm_after = g->memory->ida_oldnrm;
    valid_vectors = delta && weights
        && N_VGetVectorID(delta) == SUNDIALS_NVEC_SERIAL
        && N_VGetVectorID(weights) == SUNDIALS_NVEC_SERIAL
        && N_VGetLength_Serial(delta) == g->size
        && N_VGetLength_Serial(weights) == g->size;
    row.norm_observed = valid_vectors;
    if (valid_vectors) {
        row.norm = N_VWrmsNorm(delta, weights);
        row.norm_valid = isfinite(row.norm) && row.norm >= 0;
    }
    if (row.default_status == SUN_SUCCESS) {
        if (row.iteration_status != SUN_SUCCESS || row.iteration < 0)
            row.returned_status = SL_GUARD_OWNER;
        else if (row.iteration == 0) {
            if (!row.norm_valid || !isfinite(tol) || tol <= 0)
                row.returned_status = SL_GUARD_VALUE;
            else if (row.norm > tol) {
                row.returned_status = SUN_NLS_CONTINUE;
                row.override = 1;
            }
        }
    }
    if (g->count < g->capacity) {
        g->records[g->count++] = row;
    } else {
        /* Retain the complete stored prefix and the first missing row.
         * An earlier default failure wins over the trace-cap failure.
         */
        if (row.default_status == SUN_SUCCESS)
            row.returned_status = SL_GUARD_TRACE_FULL;
        if (!g->overflow) g->first_unrecorded = row;
        g->overflow = 1;
    }
    return row.returned_status;
}

static int sl_ida75_guard_check(sl_ida75_guard *g, void *memory, long generation)
{
    SUNNonlinearSolverContent_Newton content;
    if (!g || !memory || g->memory != (IDAMem)memory || generation != g->generation
        || g->memory->NLS != g->nls || !g->memory->ownNLS || !g->nls
        || !g->nls->ops || g->nls->ops->solve != SUNNonlinSolSolve_Newton
        || g->nls->ops->getcuriter != SUNNonlinSolGetCurIter_Newton
        || g->nls->ops->setctestfn != SUNNonlinSolSetConvTestFn_Newton
        || !g->nls->content) return SL_GUARD_OWNER;
    content = (SUNNonlinearSolverContent_Newton)g->nls->content;
    return content->CTest == sl_ida75_guard_callback && content->ctest_data == g
        && g->original_data == memory ? 0 : SL_GUARD_OWNER;
}

static int sl_ida75_guard_install(void *memory, sunindextype size, long generation,
                                 size_t capacity, sl_ida75_guard **out)
{
    IDAMem m;
    SUNNonlinearSolver nls;
    SUNNonlinearSolverContent_Newton content;
    sl_ida75_guard *g;
    Dl_info callback_image, ida_image;
    int major, minor, patch, status;
    char label[128];
    long steps;
    sunrealtype current_time;
    if (!out || *out || !memory || size < 1 || generation < 1
        || capacity < 1 || capacity > SL_GUARD_MAX_RECORDS) return SL_GUARD_OWNER;
    status = SUNDIALSGetVersionNumber(&major, &minor, &patch, label, sizeof(label));
    if (status || major != 7 || minor != 5 || patch != 0) return SL_OBS_VERSION;
    if (IDAGetNumSteps(memory, &steps) || IDAGetCurrentTime(memory, &current_time))
        return SL_OBS_GETTER;
    m = (IDAMem)memory;
    if (!m->ida_MallocDone || steps != m->ida_nst
        || !sl_ida75_same_word(current_time, m->ida_tn) || !m->ownNLS)
        return SL_GUARD_OWNER;
    nls = m->NLS;
    if (!nls || !nls->ops || !nls->content
        || nls->ops->solve != SUNNonlinSolSolve_Newton
        || nls->ops->getcuriter != SUNNonlinSolGetCurIter_Newton
        || nls->ops->setctestfn != SUNNonlinSolSetConvTestFn_Newton)
        return SL_GUARD_OWNER;
    content = (SUNNonlinearSolverContent_Newton)nls->content;
    if (!content->delta || N_VGetVectorID(content->delta) != SUNDIALS_NVEC_SERIAL
        || N_VGetLength_Serial(content->delta) != size
        || !content->CTest || content->ctest_data != memory
        || !dladdr((const void *)content->CTest, &callback_image)
        || !dladdr((const void *)&IDAGetCurrentTime, &ida_image)
        || callback_image.dli_fbase != ida_image.dli_fbase) return SL_GUARD_OWNER;
    g = (sl_ida75_guard *)calloc(1, sizeof(*g));
    if (!g) return SL_GUARD_ALLOCATION;
    g->records = (sl_ida75_guard_record *)calloc(capacity, sizeof(*g->records));
    if (!g->records) { free(g); return SL_GUARD_ALLOCATION; }
    g->memory = m;
    g->nls = nls;
    g->original = content->CTest;
    g->original_data = content->ctest_data;
    g->size = size;
    g->generation = generation;
    g->capacity = capacity;
    *out = g;  /* Failed installation is released after the native consumer. */
    status = SUNNonlinSolSetConvTestFn(nls, sl_ida75_guard_callback, g);
    if (status) return status;
    return sl_ida75_guard_check(g, memory, generation);
}

static int sl_ida75_guard_reinit(sl_ida75_guard *g, void *memory, long generation)
{
    int status;
    if (!g) return 0;
    status = sl_ida75_guard_check(g, memory, g->generation);
    if (status || g->active || generation <= g->generation) return SL_GUARD_OWNER;
    g->generation = generation;
    g->operation = g->calls = 0;
    g->count = 0;
    g->overflow = g->oldnrm_valid = 0;
    return 0;
}

static int sl_ida75_guard_begin(sl_ida75_guard *g, void *memory, long generation)
{
    int status;
    if (!g) return 0;
    status = sl_ida75_guard_check(g, memory, generation);
    if (status || g->active) return SL_GUARD_OWNER;
    g->operation++;
    g->calls = 0;
    g->count = 0;
    g->overflow = 0;
    g->active = 1;
    return 0;
}

static void sl_ida75_guard_end(sl_ida75_guard *g, int native_status)
{
    if (g) { g->native_status = native_status; g->active = 0; }
}

/* Call only after IDAFree has destroyed the consumer. No borrowed pointer is
 * dereferenced during guard release, including partial-setup cleanup.
 */
static void sl_ida75_guard_free(sl_ida75_guard **guard)
{
    if (guard && *guard) {
        free((*guard)->records);
        free(*guard);
        *guard = NULL;
    }
}

static uint64_t sl_ida75_guard_word(sunrealtype value)
{
    uint64_t word;
    memcpy(&word, &value, sizeof(word));
    return word;
}
#endif
