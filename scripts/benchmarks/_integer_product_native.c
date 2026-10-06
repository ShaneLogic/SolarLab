/* Exact a*b == high+low for finite builtin binary64 values.
 * No floating arithmetic: memcpy decodes stored words; all work is integer.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <float.h>
#include <limits.h>
#include <stdint.h>
#include <string.h>

#ifndef INTEGER_SOURCE_SHA256
#define INTEGER_SOURCE_SHA256 "unbound"
#endif
#ifndef INTEGER_BUILD_ID
#define INTEGER_BUILD_ID "unbound"
#endif

#if defined(__SIZEOF_INT128__) && __SIZEOF_INT128__ == 16
#define INTEGER_BITS 128
typedef __int128 signed128;
typedef unsigned __int128 unsigned128;
typedef struct { signed128 coefficient; int exponent; } Dyadic;

/* Call only for a nonzero magnitude: each shift is in [0, 106]. */
static Dyadic
normalize(signed128 coefficient, int exponent)
{
    unsigned128 magnitude = coefficient < 0
        ? (unsigned128)(-coefficient) : (unsigned128)coefficient;
    uint64_t lower = (uint64_t)magnitude;
    unsigned shift = lower ? (unsigned)__builtin_ctzll((unsigned long long)lower)
        : 64U + (unsigned)__builtin_ctzll((unsigned long long)(magnitude >> 64));
    magnitude >>= shift;
    Dyadic result = {coefficient < 0 ? -(signed128)magnitude
                                    : (signed128)magnitude,
                     exponent + (int)shift};
    return result;
}

static int
decode(PyObject *value, Dyadic *result)
{
    uint64_t bits;
    if (!PyFloat_CheckExact(value))
        return 0;
    memcpy(&bits, &((PyFloatObject *)value)->ob_fval, sizeof bits);
    unsigned raw_exponent = (unsigned)((bits >> 52) & UINT64_C(0x7ff));
    if (raw_exponent == 0x7ffU)
        return 0;
    uint64_t coefficient = bits & UINT64_C(0x000fffffffffffff);
    if (raw_exponent)
        coefficient |= UINT64_C(0x0010000000000000);
    if (!coefficient) {
        *result = (Dyadic){0, 0};  /* Both zero signs denote the same real zero. */
        return 1;
    }
    signed128 signed_coefficient = (signed128)coefficient;
    if (bits >> 63)
        signed_coefficient = -signed_coefficient;
    *result = normalize(signed_coefficient,
                        raw_exponent ? (int)raw_exponent - 1075 : -1074);
    return 1;
}

static int
zero_sum(Dyadic terms[3])
{
    Dyadic nonzero[3];
    int count = 0;
    for (int i = 0; i < 3; ++i)
        if (terms[i].coefficient)
            nonzero[count++] = terms[i];
    if (!count)
        return 1;
    if (count == 1)
        return 0;
    if (count == 2)
        return nonzero[0].exponent == nonzero[1].exponent
            && nonzero[0].coefficient == -nonzero[1].coefficient;

    int minimum = nonzero[0].exponent;
    for (int i = 1; i < 3; ++i)
        if (nonzero[i].exponent < minimum)
            minimum = nonzero[i].exponent;
    int first = -1, second = -1, third = -1;
    for (int i = 0; i < 3; ++i) {
        if (nonzero[i].exponent != minimum)
            third = i;
        else if (first < 0)
            first = i;
        else if (second < 0)
            second = i;
        else
            return 0;  /* Three odd coefficients at one exponent sum to odd. */
    }
    if (second < 0)
        return 0;  /* A unique odd minimum cannot cancel two even multiples. */
    signed128 sum = nonzero[first].coefficient + nonzero[second].coefficient;
    if (!sum)
        return 0;  /* The remaining third coefficient is nonzero. */
    Dyadic combined = normalize(sum, minimum);
    return combined.exponent == nonzero[third].exponent
        && combined.coefficient == -nonzero[third].coefficient;
}
#else
#define INTEGER_BITS 0
#endif

static int supported_abi;

static PyObject *
matches(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    (void)self;
    if (nargs != 4) {
        PyErr_SetString(PyExc_TypeError, "matches requires four positional values");
        return NULL;
    }
#if INTEGER_BITS == 128
    if (supported_abi) {
        Dyadic a, b, high, low;
        if (!decode(args[0], &a) || !decode(args[1], &b)
                || !decode(args[2], &high) || !decode(args[3], &low))
            Py_RETURN_NOTIMPLEMENTED;
        /* Operand coefficients have <=53 bits; their product has <=106.
         * Every later signed addition has <=107 bits, strictly below 127.
         * Exponents lie in [-2148, 2153], and no large alignment shift occurs.
         */
        Dyadic terms[3] = {
            {a.coefficient * b.coefficient, a.exponent + b.exponent},
            {-high.coefficient, high.exponent},
            {-low.coefficient, low.exponent}
        };
        return PyBool_FromLong(zero_sum(terms));
    }
#endif
    Py_RETURN_NOTIMPLEMENTED;
}

static PyMethodDef methods[] = {
    {"matches", _PyCFunction_CAST(matches), METH_FASTCALL,
     "Exact finite builtin-float equality; NotImplemented outside the ABI/types."},
    {NULL, NULL, 0, NULL}
};
static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_integer_product_native", NULL, -1, methods,
    NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC
PyInit__integer_product_native(void)
{
    supported_abi = CHAR_BIT == 8 && sizeof(double) == 8 && sizeof(uint64_t) == 8
        && sizeof(unsigned long long) == 8 && sizeof(int) >= 4
        && FLT_RADIX == 2 && DBL_MANT_DIG == 53
        && DBL_MIN_EXP == -1021 && DBL_MAX_EXP == 1024 && INTEGER_BITS == 128;
#if !defined(__APPLE__) || !(defined(__arm64__) || defined(__aarch64__))
    supported_abi = 0;  /* This optional build attests only Darwin arm64. */
#endif
    if (supported_abi) {
        /* Exact stored constants, never evaluated as arithmetic or compared
         * as floats; these establish the double/uint64 bit-layout agreement.
         */
        static const double values[] = {
            0.0, -0.0, 1.0, 0x1.0000000000001p0,
            0x1p-1022, 0x0.0000000000001p-1022, 0x1.fffffffffffffp1023
        };
        static const uint64_t expected[] = {
            UINT64_C(0), UINT64_C(0x8000000000000000),
            UINT64_C(0x3ff0000000000000), UINT64_C(0x3ff0000000000001),
            UINT64_C(0x0010000000000000), UINT64_C(1),
            UINT64_C(0x7fefffffffffffff)
        };
        for (unsigned i = 0; i < sizeof values / sizeof values[0]; ++i) {
            uint64_t bits;
            memcpy(&bits, &values[i], sizeof bits);
            if (bits != expected[i])
                supported_abi = 0;
        }
    }
    PyObject *result = PyModule_Create(&module);
    if (!result)
        return NULL;
    PyObject *abi = Py_BuildValue("{s:O,s:i,s:i,s:i,s:i,s:i,s:i,s:s,s:s}",
        "supported", supported_abi ? Py_True : Py_False,
        "char_bits", CHAR_BIT, "double_bytes", (int)sizeof(double),
        "uint64_bytes", (int)sizeof(uint64_t), "integer_bits", INTEGER_BITS,
        "mantissa_bits", DBL_MANT_DIG, "layout_witnesses", 7,
        "python_version", PY_VERSION, "compiler", __VERSION__);
    if (!abi || PyModule_AddObject(result, "ABI", abi) < 0) {
        Py_XDECREF(abi);
        Py_DECREF(result);
        return NULL;
    }
    if (PyModule_AddStringConstant(result, "SOURCE_SHA256", INTEGER_SOURCE_SHA256) < 0
            || PyModule_AddStringConstant(result, "BUILD_ID", INTEGER_BUILD_ID) < 0) {
        Py_DECREF(result);
        return NULL;
    }
    return result;
}
