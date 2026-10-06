/* Source-owned lifetime entry, exact SUNDIALS 7.5.0 double/int32 ABI. */
#ifndef SOLARLAB_IDA_SUPERLUMT_CLEANUP_H
#define SOLARLAB_IDA_SUPERLUMT_CLEANUP_H
#include <sunlinsol/sunlinsol_superlumt.h>
#if SUNDIALS_VERSION_MAJOR != 7 || SUNDIALS_VERSION_MINOR != 5 || SUNDIALS_VERSION_PATCH != 0
#error "The cleanup source requires SUNDIALS 7.5.0"
#endif
#if !defined(SUNDIALS_DOUBLE_PRECISION) || !defined(SUNDIALS_INT32_T)
#error "The cleanup source requires the pinned double/int32 configuration"
#endif
SUNLinearSolver sl_SUNLinSol_SuperLUMT(N_Vector y, SUNMatrix A, int num_threads, SUNContext sunctx);
#endif
