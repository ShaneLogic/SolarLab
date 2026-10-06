/* -----------------------------------------------------------------
 * Programmer(s): Daniel Reynolds @ UMBC
 * -----------------------------------------------------------------
 * Based on codes <solver>_superlumt.c, written by
 * Carol S. Woodward @ LLNL
 * -----------------------------------------------------------------
 * SUNDIALS Copyright Start
 * Copyright (c) 2025, Lawrence Livermore National Security,
 * University of Maryland Baltimore County, and the SUNDIALS contributors.
 * Copyright (c) 2013-2025, Lawrence Livermore National Security
 * and Southern Methodist University.
 * Copyright (c) 2002-2013, Lawrence Livermore National Security.
 * All rights reserved.
 *
 * See the top-level LICENSE and NOTICE files for details.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * SUNDIALS Copyright End
 * -----------------------------------------------------------------
 * This is the implementation file for the SuperLUMT implementation
 * of the SUNLINSOL package.
 * -----------------------------------------------------------------*/

/* Source-owned constructor, factor retirement and free, derived from the pinned 7.5.0 file.
 * All factorization, solve, ordering and statistics routines remain in the
 * unchanged linked SUNDIALS/SuperLU_MT library. See SourcePinsV1.json. */
#include <stdlib.h>
#include "ida_superlumt_cleanup.h"

#define ONE SUN_RCONST(1.0)

#define SLUMT_CONTENT(S)   ((SUNLinearSolverContent_SuperLUMT)(S->content))
#define LASTFLAG(S)        (SLUMT_CONTENT(S)->last_flag)
#define FIRSTFACTORIZE(S)  (SLUMT_CONTENT(S)->first_factorize)
#define SM_A(S)            (SLUMT_CONTENT(S)->A)
#define SM_AC(S)           (SLUMT_CONTENT(S)->AC)
#define SM_L(S)            (SLUMT_CONTENT(S)->L)
#define SM_U(S)            (SLUMT_CONTENT(S)->U)
#define SM_B(S)            (SLUMT_CONTENT(S)->B)
#define GSTAT(S)           (SLUMT_CONTENT(S)->Gstat)
#define PERMR(S)           (SLUMT_CONTENT(S)->perm_r)
#define PERMC(S)           (SLUMT_CONTENT(S)->perm_c)
#define SIZE(S)            (SLUMT_CONTENT(S)->N)
#define NUMTHREADS(S)      (SLUMT_CONTENT(S)->num_threads)
#define DIAGPIVOTTHRESH(S) (SLUMT_CONTENT(S)->diag_pivot_thresh)
#define ORDERING(S)        (SLUMT_CONTENT(S)->ordering)
#define OPTIONS(S)         (SLUMT_CONTENT(S)->options)

static SUNErrCode sl_SUNLinSolFree_SuperLUMT(SUNLinearSolver S);
static SUNErrCode sl_SUNLinSolInitialize_SuperLUMT(SUNLinearSolver S);
extern SUNErrCode SUNLinSolSetOptions_SuperLUMT(SUNLinearSolver S, const char* LSid,
                                               const char* file_name, int argc, char* argv[]);

SUNLinearSolver sl_SUNLinSol_SuperLUMT(N_Vector y, SUNMatrix A, int num_threads,
                                    SUNContext sunctx)
{
  SUNLinearSolver S;
  SUNLinearSolverContent_SuperLUMT content;
  sunindextype MatrixRows;

  if (y == NULL || A == NULL) { return NULL; }

  /* Check compatibility with supplied SUNMatrix and N_Vector */
  if (SUNMatGetID(A) != SUNMATRIX_SPARSE) { return (NULL); }

  if (SUNSparseMatrix_Rows(A) != SUNSparseMatrix_Columns(A)) { return (NULL); }

  if ((N_VGetVectorID(y) != SUNDIALS_NVEC_SERIAL) &&
      (N_VGetVectorID(y) != SUNDIALS_NVEC_OPENMP) &&
      (N_VGetVectorID(y) != SUNDIALS_NVEC_PTHREADS))
  {
    return (NULL);
  }

  MatrixRows = SUNSparseMatrix_Rows(A);
  if (MatrixRows != N_VGetLength(y)) { return (NULL); }

  /* Create an empty linear solver */
  S = NULL;
  S = SUNLinSolNewEmpty(sunctx);
  if (S == NULL) { return (NULL); }

  /* Attach operations */
  S->ops->gettype    = SUNLinSolGetType_SuperLUMT;
  S->ops->getid      = SUNLinSolGetID_SuperLUMT;
  S->ops->initialize = sl_SUNLinSolInitialize_SuperLUMT;
  S->ops->setoptions = SUNLinSolSetOptions_SuperLUMT;
  S->ops->setup      = SUNLinSolSetup_SuperLUMT;
  S->ops->solve      = SUNLinSolSolve_SuperLUMT;
  S->ops->lastflag   = SUNLinSolLastFlag_SuperLUMT;
  S->ops->space      = SUNLinSolSpace_SuperLUMT;
  S->ops->free       = sl_SUNLinSolFree_SuperLUMT;

  /* Create content */
  content = NULL;
  content = (SUNLinearSolverContent_SuperLUMT)malloc(sizeof *content);
  if (content == NULL)
  {
    SUNLinSolFree(S);
    return (NULL);
  }

  /* Attach content */
  S->content = content;

  /* Fill content */
  content->N                 = MatrixRows;
  content->last_flag         = 0;
  content->num_threads       = num_threads;
  content->diag_pivot_thresh = ONE;
  content->ordering          = SUNSLUMT_ORDERING_DEFAULT;
  content->perm_r            = NULL;
  content->perm_c            = NULL;
  content->Gstat             = NULL;
  content->A                 = NULL;
  content->AC                = NULL;
  content->L                 = NULL;
  content->U                 = NULL;
  content->B                 = NULL;
  content->options           = NULL;

  /* Allocate content */
  content->perm_r = (sunindextype*)malloc(MatrixRows * sizeof(sunindextype));
  if (content->perm_r == NULL)
  {
    SUNLinSolFree(S);
    return (NULL);
  }

  content->perm_c = (sunindextype*)malloc(MatrixRows * sizeof(sunindextype));
  if (content->perm_c == NULL)
  {
    SUNLinSolFree(S);
    return (NULL);
  }

  content->Gstat = (Gstat_t*)calloc(1, sizeof(Gstat_t));
  if (content->Gstat == NULL)
  {
    SUNLinSolFree(S);
    return (NULL);
  }

  content->A = (SuperMatrix*)malloc(sizeof(SuperMatrix));
  if (content->A == NULL)
  {
    SUNLinSolFree(S);
    return (NULL);
  }
  content->A->Store = NULL;

  content->AC = (SuperMatrix*)malloc(sizeof(SuperMatrix));
  if (content->AC == NULL)
  {
    SUNLinSolFree(S);
    return (NULL);
  }
  content->AC->Store = NULL;

  content->L = (SuperMatrix*)malloc(sizeof(SuperMatrix));
  if (content->L == NULL)
  {
    SUNLinSolFree(S);
    return (NULL);
  }
  content->L->Store = NULL;

  content->U = (SuperMatrix*)malloc(sizeof(SuperMatrix));
  if (content->U == NULL)
  {
    SUNLinSolFree(S);
    return (NULL);
  }
  content->U->Store = NULL;

  content->B = (SuperMatrix*)malloc(sizeof(SuperMatrix));
  if (content->B == NULL)
  {
    SUNLinSolFree(S);
    return (NULL);
  }
  content->B->Store = NULL;
  xCreate_Dense_Matrix(content->B, MatrixRows, 1, NULL, MatrixRows, SLU_DN,
                       SLU_D, SLU_GE);

  content->options = (superlumt_options_t*)calloc(1, sizeof(superlumt_options_t));
  if (content->options == NULL)
  {
    SUNLinSolFree(S);
    return (NULL);
  }
  StatAlloc(MatrixRows, num_threads, sp_ienv(1), sp_ienv(2), content->Gstat);

  return (S);
}

/* Retire only factor-owned data. Matrix wrappers, permutations, options and
   Gstat remain owned by the solver. Native initialize resets FIRSTFACTORIZE,
   so every subsequent refact=NO setup must begin without the previous data. */
static void sl_RetireFactorData_SuperLUMT(SUNLinearSolver S)
{
  if (OPTIONS(S))
  {
    SUPERLU_FREE(OPTIONS(S)->etree);
    OPTIONS(S)->etree = NULL;
    SUPERLU_FREE(OPTIONS(S)->colcnt_h);
    OPTIONS(S)->colcnt_h = NULL;
    SUPERLU_FREE(OPTIONS(S)->part_super_h);
    OPTIONS(S)->part_super_h = NULL;
  }
  if (SM_AC(S) && SM_AC(S)->Store)
  {
    Destroy_CompCol_Permuted(SM_AC(S));
    SM_AC(S)->Store = NULL;
  }
  if (SM_L(S) && SM_L(S)->Store)
  {
    Destroy_SuperNode_SCP(SM_L(S));
    SM_L(S)->Store = NULL;
  }
  if (SM_U(S) && SM_U(S)->Store)
  {
    Destroy_CompCol_NCP(SM_U(S));
    SM_U(S)->Store = NULL;
  }
}

static SUNErrCode sl_SUNLinSolInitialize_SuperLUMT(SUNLinearSolver S)
{
  if (S == NULL || S->content == NULL) { return SUN_ERR_ARG_CORRUPT; }
  sl_RetireFactorData_SuperLUMT(S);
  return SUNLinSolInitialize_SuperLUMT(S);
}

static SUNErrCode sl_SUNLinSolFree_SuperLUMT(SUNLinearSolver S)
{
  /* return with success if already freed */
  if (S == NULL) { return SUN_SUCCESS; }

  /* delete items from the contents structure (if it exists) */
  if (S->content)
  {
    sl_RetireFactorData_SuperLUMT(S);
    if (PERMR(S))
    {
      free(PERMR(S));
      PERMR(S) = NULL;
    }
    if (PERMC(S))
    {
      free(PERMC(S));
      PERMC(S) = NULL;
    }
    if (OPTIONS(S))
    {
      free(OPTIONS(S));
      OPTIONS(S) = NULL;
    }
    if (SM_L(S))
    {
      free(SM_L(S));
      SM_L(S) = NULL;
    }
    if (SM_U(S))
    {
      free(SM_U(S));
      SM_U(S) = NULL;
    }
    if (GSTAT(S))
    {
      StatFree(GSTAT(S));
      free(GSTAT(S));
      GSTAT(S) = NULL;
    }
    if (SM_B(S))
    {
      if (SM_B(S)->Store)
      {
        Destroy_SuperMatrix_Store(SM_B(S));
        SM_B(S)->Store = NULL;
      }
      free(SM_B(S));
      SM_B(S) = NULL;
    }
    if (SM_A(S))
    {
      if (SM_A(S)->Store)
      {
        SUPERLU_FREE(SM_A(S)->Store);
        SM_A(S)->Store = NULL;
      }
      free(SM_A(S));
      SM_A(S) = NULL;
    }
    if (SM_AC(S))
    {
      free(SM_AC(S));
      SM_AC(S) = NULL;
    }
    free(S->content);
    S->content = NULL;
  }

  /* delete generic structures */
  if (S->ops)
  {
    free(S->ops);
    S->ops = NULL;
  }
  free(S);
  S = NULL;
  return SUN_SUCCESS;
}
