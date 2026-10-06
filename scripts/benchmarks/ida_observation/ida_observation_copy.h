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
#include <string.h>
#include <nvector/nvector_serial.h>
#include <sundials/sundials_version.h>
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
#endif
