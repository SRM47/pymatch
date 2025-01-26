#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include <stddef.h>

#include "tensorbase.h"

/*********************************************************
 *               PyTensorBase Definition                 *
 *********************************************************/

// A wrapper around TensorBase enabling use as a Python object.
// We try to match the pytorch tensor API as closely as possible.
// clang-format off
typedef struct
{
    PyObject_HEAD
    TensorBase tb;
} PyTensorBase;
// clang-format on

/*********************************************************
 *            Initialization and Deallocation            *
 *********************************************************/

// __init__
static int PyTensorBase_init(PyTensorBase *self, PyObject *args, PyObject *kwds);
// free / Deallocation.
static void PyTensorBase_dealloc(PyTensorBase *self);

/*********************************************************
 *                   Number Protocol                     *
 *********************************************************/
static PyObject *PyTensorBase_matrix_multiply(PyObject *a, PyObject *b);

static PyObject *PyTensorBase_nb_add(PyObject *a, PyObject *b);
static PyObject *PyTensorBase_nb_subtract(PyObject *a, PyObject *b);
static PyObject *PyTensorBase_nb_multiply(PyObject *a, PyObject *b);
static PyObject *PyTensorBase_nb_floor_divide(PyObject *a, PyObject *b);
static PyObject *PyTensorBase_nb_true_divide(PyObject *a, PyObject *b);
static PyObject *PyTensorBase_nb_power(PyObject *a, PyObject *b, PyObject *c);
static PyObject *PyTensorBase_nb_negative(PyObject *a);
static PyObject *PyTensorBase_nb_positive(PyObject *a);
static PyObject *PyTensorBase_nb_absolute(PyObject *a);

// https://docs.python.org/3/c-api/typeobj.html#number-object-structures
static PyNumberMethods PyTensorBase_as_number = {
    .nb_add = (binaryfunc)PyTensorBase_nb_add,
    .nb_subtract = (binaryfunc)PyTensorBase_nb_subtract,
    .nb_multiply = (binaryfunc)PyTensorBase_nb_multiply,
    .nb_floor_divide = (binaryfunc)PyTensorBase_nb_floor_divide,
    .nb_true_divide = (binaryfunc)PyTensorBase_nb_true_divide,
    // .nb_remainder = 0,
    // .nb_divmod = 0,
    .nb_matrix_multiply = (binaryfunc)PyTensorBase_matrix_multiply,
    .nb_power = (ternaryfunc)PyTensorBase_nb_power,
    .nb_negative = (unaryfunc)PyTensorBase_nb_negate,
    .nb_positive = (unaryfunc)PyTensorBase_nb_positive,
    .nb_absolute = (unaryfunc)PyTensorBase_nb_absolute,
    // .nb_invert = 0,
    // .nb_lshift = 0,
    // .nb_rshift = 0,
    // .nb_bool = 0,
    // .nb_and = 0,
    // .nb_xor = 0,
    // .nb_or = 0,
    // .nb_int = 0,
    // .nb_float = 0,
    // .nb_inplace_add = 0,
    // .nb_inplace_subtract = 0,
    // .nb_inplace_multiply = 0,
    // .nb_inplace_floor_divide = 0,
    // .nb_inplace_true_divide = 0,
    // .nb_inplace_remainder = 0,
    // .nb_inplace_matrix_multiply = 0,
    // .nb_inplace_power = 0,
    // .nb_inplace_lshift = 0,
    // .nb_inplace_rshift = 0,
    // .nb_inplace_and = 0,
    // .nb_inplace_xor = 0,
    // .nb_inplace_or = 0,
    // .nb_index = 0,
};

/*********************************************************
 *                  Sequence Protocol                    *
 *********************************************************/

// https://docs.python.org/3/c-api/typeobj.html#sequence-object-structures
static PySequenceMethods PyTensorBase_sequence_methods = {
    // .sq_length = 0,
    // .sq_concat = 0,
    // .sq_repeat = 0,
    // .sq_item = 0,
    // .sq_ass_item = 0,
    // .sq_contains = 0,
    // .sq_inplace_concat = 0,
    // .sq_inplace_repeat = 0,
};

/*********************************************************
 *                   Mapping Methods                     *
 *********************************************************/

// https://docs.python.org/3/c-api/typeobj.html#mapping-object-structures
static PyMappingMethods PyTensorBase_mapping_methods = {
    .mp_length = 0,
    .mp_subscript = 0,
    .mp_ass_subscript = 0,
};

/*********************************************************
 *                   Instance Methods                    *
 *********************************************************/
// Excludes all number methods.

static PyObject *PyTensorBase_abs_(PyObject *self, PyObject *Py_UNUSED(args));
static PyObject *PyTensorBase_abs(PyObject *self, PyObject *Py_UNUSED(args));

static PyObject *PyTensorBase_cos_(PyObject *self, PyObject *Py_UNUSED(args));
static PyObject *PyTensorBase_cos(PyObject *self, PyObject *Py_UNUSED(args));

static PyObject *PyTensorBase_sin_(PyObject *self, PyObject *Py_UNUSED(args));
static PyObject *PyTensorBase_sin(PyObject *self, PyObject *Py_UNUSED(args));

static PyObject *PyTensorBase_tan_(PyObject *self, PyObject *Py_UNUSED(args));
static PyObject *PyTensorBase_tan(PyObject *self, PyObject *Py_UNUSED(args));

static PyObject *PyTensorBase_tanh_(PyObject *self, PyObject *Py_UNUSED(args));
static PyObject *PyTensorBase_tanh(PyObject *self, PyObject *Py_UNUSED(args));

static PyObject *PyTensorBase_log_(PyObject *self, PyObject *Py_UNUSED(args));
static PyObject *PyTensorBase_log(PyObject *self, PyObject *Py_UNUSED(args));

static PyObject *PyTensorBase_exp_(PyObject *self, PyObject *Py_UNUSED(args));
static PyObject *PyTensorBase_exp(PyObject *self, PyObject *Py_UNUSED(args));

static PyObject *PyTensorBase_sigmoid_(PyObject *self, PyObject *Py_UNUSED(args));
static PyObject *PyTensorBase_sigmoid(PyObject *self, PyObject *Py_UNUSED(args));

static PyObject *PyTensorBase_zero_(PyObject *self, PyObject *Py_UNUSED(args));

static PyObject *PyTensorBase_reshape_(PyObject *self, PyObject *args);
static PyObject *PyTensorBase_reshape(PyObject *self, PyObject *args);

static PyObject *PyTensorBase_fill_(PyObject *self, PyObject *args);

static PyObject *PyTensorBase_max(PyObject *self, PyObject *const *args, Py_ssize_t nargs);
static PyObject *PyTensorBase_min(PyObject *self, PyObject *const *args, Py_ssize_t nargs);
static PyObject *PyTensorBase_mean(PyObject *self, PyObject *const *args, Py_ssize_t nargs);
static PyObject *PyTensorBase_sum(PyObject *self, PyObject *const *args, Py_ssize_t nargs);

static PyObject *PyTensorBase_broadcast_to(PyObject *self, PyObject *args);
static PyObject *PyTensorBase_permute(PyObject *self, PyObject *args);
static PyObject *PyTensorBase_transpose(PyObject *self, PyObject *args);

static PyMethodDef PyTensorBase_instance_methods[] = {
    // Methods with no arguments.
    {"abs_", (PyCFunction)PyTensorBase_abs_, METH_NOARGS, "In-place absolute value."},
    {"abs", (PyCFunction)PyTensorBase_abs, METH_NOARGS, "Out-of-place absolute value."},

    {"cos_", (PyCFunction)PyTensorBase_cos_, METH_NOARGS, "In-place cosine."},
    {"cos", (PyCFunction)PyTensorBase_cos, METH_NOARGS, "Out-of-place cosine."},

    {"sin_", (PyCFunction)PyTensorBase_sin_, METH_NOARGS, "In-place sine."},
    {"sin", (PyCFunction)PyTensorBase_sin, METH_NOARGS, "Out-of-place sine."},

    {"tan_", (PyCFunction)PyTensorBase_tan_, METH_NOARGS, "In-place tangent."},
    {"tan", (PyCFunction)PyTensorBase_tan, METH_NOARGS, "Out-of-place tangent."},

    {"tanh_", (PyCFunction)PyTensorBase_tanh_, METH_NOARGS, "In-place hyperbolic tangent."},
    {"tanh", (PyCFunction)PyTensorBase_tanh, METH_NOARGS, "Out-of-place hyperbolic tangent."},

    {"log_", (PyCFunction)PyTensorBase_log_, METH_NOARGS, "In-place natural logarithm."},
    {"log", (PyCFunction)PyTensorBase_log, METH_NOARGS, "Out-of-place natural logarithm."},

    {"exp_", (PyCFunction)PyTensorBase_exp_, METH_NOARGS, "In-place exponential."},
    {"exp", (PyCFunction)PyTensorBase_exp, METH_NOARGS, "Out-of-place exponential."},

    {"sigmoid_", (PyCFunction)PyTensorBase_sigmoid_, METH_NOARGS, "In-place sigmoid."},
    {"sigmoid", (PyCFunction)PyTensorBase_sigmoid, METH_NOARGS, "Out-of-place sigmoid."},

    {"zero_", (PyCFunction)PyTensorBase_zero_, METH_NOARGS, "In-place zero."},

    // Methods with arguments.
    {"reshape_", (PyCFunction)PyTensorBase_reshape_, METH_O, "In-place reshape."},
    {"reshape", (PyCFunction)PyTensorBase_reshape, METH_O, "Out-of-place reshape."},

    {"fill_", (PyCFunction)PyTensorBase_fill_, METH_O, "In-place fill."},

    {"max", (PyCFunctionFast)PyTensorBase_max, METH_FASTCALL, "Compute the maximum value."},
    {"min", (PyCFunctionFast)PyTensorBase_min, METH_FASTCALL, "Compute the minimum value."},

    {"mean", (PyCFunctionFast)PyTensorBase_mean, METH_FASTCALL, "Compute the mean value."},
    {"sum", (PyCFunctionFast)PyTensorBase_sum, METH_FASTCALL, "Compute the sum of elements."},

    {"broadcast_to", (PyCFunction)PyTensorBase_broadcast_to, METH_O, "Broadcast the array to a new shape."},
    {"permute", (PyCFunction)PyTensorBase_permute, METH_O, "Permute the dimensions of the array."},
    {"transpose", (PyCFunction)PyTensorBase_transpose, METH_O, "Transpose the array."},
    {NULL} /* Sentinel */
};

/*********************************************************
 *               Member Variable Definition              *
 *********************************************************/
// Expose read-only instance variables.
static PyMemberDef PyTensorBase_members[] = {
    {"ndim", Py_T_LONG, offsetof(PyTensorBase, tb.ndim), Py_READONLY,
     "Tensor rank"},
    {NULL} /* Sentinel */
};

/*********************************************************
 *                  Get Set Definition                   *
 *********************************************************/
static PyObject *PyTensorBase_get_dim(PyTensorBase *self, PyObject *Py_UNUSED(ignored));
static PyObject *PyTensorBase_get_size(PyTensorBase *self, PyObject *Py_UNUSED(ignored));
static PyObject *PyTensorBase_get_numel(PyTensorBase *self, PyObject *Py_UNUSED(ignored));
static PyObject *PyTensorBase_get_stride(PyTensorBase *self, PyObject *Py_UNUSED(ignored));

static PyGetSetDef PyTensorBase_getset[] = {
    {"dim", (getter)PyTensorBase_get_dim, NULL, "Gets tensor rank", NULL},
    {"size", (getter)PyTensorBase_get_size, NULL, "Tensor shape", NULL},
    {"numel", (getter)PyTensorBase_get_size, NULL, "Number of elements in Tensor", NULL},
    {"stride", (getter)PyTensorBase_get_size, NULL, "Strides of tensor", NULL},
    {NULL} /* Sentinel */
};

/*********************************************************
 *                 Class/Module Methods                  *
 *********************************************************/

static PyObject *PyTensorBase_ones(PyModuleDef *module, PyObject *args);
static PyObject *PyTensorBase_randn(PyModuleDef *module, PyObject *args);

static PyMethodDef PyTensorBase_class_methods[] = {
    {"ones", (PyCFunction)PyTensorBase_ones, METH_VARARGS, "TODO: docs"},
    {"randn", (PyCFunction)PyTensorBase_randn, METH_VARARGS, "TODO: docs"},
    {NULL} /* Sentinel */
};

/*********************************************************
 *                   String Methods                      *
 *********************************************************/

static PyObject *PyTensorBase_str(PyTensorBase *obj);

/*********************************************************
 *                   Module Definition                   *
 *********************************************************/

static PyTypeObject PyTensorBaseType = {
    PyVarObject_HEAD_INIT(NULL, 0),
    .tp_name = "match.tensorbase.TensorBase", /* For printing, in format "<module>.<name>" */
    .tp_basicsize = sizeof(PyTensorBase),
    .tp_itemsize = 0, /* For allocation */

    /* Methods to implement standard operations */
    .tp_dealloc = (destructor)PyTensorBase_dealloc,
    // .tp_vectorcall_offset = 0,
    // .tp_getattr = 0,
    // .tp_setattr = 0,
    // .tp_as_async = 0, /* formerly known as tp_compare (Python 2) or tp_reserved (Python 3) */
    // .tp_repr = 0,

    /* Method suites for standard classes */
    .tp_as_number = &PyTensorBase_as_number,
    .tp_as_sequence = &PyTensorBase_sequence_methods,
    .tp_as_mapping = &PyTensorBase_mapping_methods,

    /* More standard operations (here for binary compatibility) */
    // .tp_hash = 0,
    // .tp_call = 0,
    .tp_str = PyTensorBase_str,
    // .tp_getattro = 0,
    // .tp_setattro = 0,

    /* Functions to access object as input/output buffer */
    // .tp_as_buffer = 0,

    /* Flags to define presence of optional/expanded features */
    .tp_flags = Py_TPFLAGS_DEFAULT,

    .tp_doc = PyDoc_STR("TODO: docs"), /* Documentation string */

    /* Assigned meaning in release 2.0 */
    /* call function for all accessible objects */
    // .tp_traverse = 0,

    /* delete references to contained objects */
    // .tp_clear = 0,

    /* Assigned meaning in release 2.1 */
    /* rich comparisons */
    // .tp_richcompare = 0,

    /* weak reference enabler */
    // .tp_weaklistoffset = 0,

    /* Iterators */
    // .tp_iter = 0,
    // .tp_iternext = 0,

    /* Attribute descriptor and subclassing stuff */
    .tp_methods = PyTensorBase_instance_methods,
    .tp_members = PyTensorBase_members,
    .tp_getset = PyTensorBase_getset,
    // Strong reference on a heap type, borrowed reference on a static type
    // .tp_base = 0,
    // .tp_dict = 0,
    // .tp_descr_get = 0,
    // .tp_descr_set = 0,
    // .tp_dictoffset = 0,
    .tp_init = (initproc)PyTensorBase_init,
    // .tp_alloc = 0,
    .tp_new = PyType_GenericNew,
    // .tp_free = 0,  /* Low-level free-memory routine */
    // .tp_is_gc = 0, /* For PyObject_IS_GC */
    // .tp_bases = 0,
    // .tp_mro = 0, /* method resolution order */
    // .tp_cache = 0,
    // .tp_subclasses = 0,
    // .tp_weaklist = 0,
    // .tp_del = 0,

    /* Type attribute cache version tag. Added in version 2.6 */
    // .tp_version_tag = 0,

    // .tp_finalize = 0,
    // .tp_vectorcall = 0,

    /* bitset of which type-watchers care about this type */
    // .tp_watched = 0,
};

static PyModuleDef TensorBaseModule = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "match.tensorbase",
    .m_doc = PyDoc_STR("TODO: docs"),
    .m_size = -1,
    .m_methods = PyTensorBase_class_methods,
};

PyMODINIT_FUNC
PyInit_tensorbase(void)
{
    if (PyType_Ready(&PyTensorBaseType) < 0)
        return NULL;

    PyObject *m = PyModule_Create(&TensorBaseModule);
    if (m == NULL)
        return NULL;

    Py_INCREF(&PyTensorBaseType);
    if (PyModule_AddObject(m, "TensorBase", (PyObject *)&PyTensorBaseType) < 0)
    {
        Py_DECREF(&PyTensorBaseType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}

/*********************************************************
 *                    IMPLEMENTATION                     *
 *********************************************************/

/*********************************************************
 *                       Utilities                       *
 *********************************************************/

static long PyTensorBase_Check(PyObject *obj)
{
    return PyObject_IsInstance(obj, (PyObject *)&PyTensorBaseType);
}

static long PyFloatOrLong_Check(PyObject *obj)
{
    return PyLong_Check(obj) || PyFloat_Check(obj);
}

static long can_math(PyObject *obj)
{
    return PyFloatOrLong_Check(obj) || PyTensorBase_Check(obj);
}

static scalar PyFloatOrLong_asDouble(PyObject *obj)
{
    if (PyLong_Check(obj))
    {
        return PyLong_AsDouble(obj);
    }
    return PyFloat_AsDouble(obj);
}

static PyTensorBase *PyTensorBase_shallow_broadcast(PyTensorBase *t, ShapeArray shape);

static PyObject *PyTensorBase_nb_binary_operation(PyObject *a, PyObject *b, scalar (*op)(scalar, scalar))
{
    PyTensorBase *result = (PyTensorBase *)PyObject_New(PyTensorBase, &PyTensorBaseType);
    if (result == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create new TensorBase object.");
        return NULL;
    }

    // PyTensorBase + (Long | Float)
    if (PyTensorBase_Check(a) && PyFloatOrLong_Check(b))
    {
        TensorBase *t = &(((PyTensorBase *)a)->tb);
        scalar s = PyFloatOrLong_asDouble(b);
        if (TensorBase_binary_op_tensorbase_scalar(t, s, &(result->tb), op) < 0)
        {
            PyErr_SetString(PyExc_RuntimeError, "Error performing binary operation");
            return NULL;
        }
    }
    // (Long | Float) + PyTensorBase
    else if (PyFloatOrLong_Check(a) && PyTensorBase_Check(b))
    {
        TensorBase *t = &(((PyTensorBase *)b)->tb);
        scalar s = PyFloatOrLong_asDouble(a);
        if (TensorBase_binary_op_scalar_tensorbase(t, s, &(result->tb), op) < 0)
        {
            PyErr_SetString(PyExc_RuntimeError, "Error performing binary operation");
            return NULL;
        }
    }
    // PyTensorBase + PyTensorBase
    else if (PyTensorBase_Check(a) && PyTensorBase_Check(b))
    {
        TensorBase *l = &(((PyTensorBase *)a)->tb);
        TensorBase *r = &(((PyTensorBase *)b)->tb);
        if (TensorBase_binary_op_tensorbase_tensorbase(l, r, &(result->tb), op) < 0)
        {
            PyErr_SetString(PyExc_RuntimeError, "Error performing binary operation");
            return NULL;
        }
    }
    // Incompatible types for mathematical binary operations
    else
    {
        PyErr_SetString(PyExc_ValueError, "Invalid types for addition.");
        return NULL;
    }
    return (PyObject *)result;
}

static PyObject *PyTensorBase_nb_unary_operation(PyObject *a, scalar (*op)(scalar, scalar))
{
    PyTensorBase *result = (PyTensorBase *)PyObject_New(PyTensorBase, &PyTensorBaseType);
    if (result == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create new TensorBase object.");
        return NULL;
    }

    TensorBase *in = &(((PyTensorBase *)a)->tb);
    if (TensorBase_unary_op(in, &(result->tb), op) < 0)
    {
        PyErr_SetString(PyExc_RuntimeError, "Error performing unary operation");
        return NULL;
    }

    return (PyObject *)result;
}

static PyObject *PyTensorBase_nb_unary_operation_inpalce(PyObject *a, scalar (*op)(scalar, scalar))
{
    // Assumes input PyObject is already of type PyTensorBase.
    TensorBase *in = &(((PyTensorBase *)a)->tb);
    if (TensorBase_unary_op_inplace(in, op) < 0)
    {
        PyErr_SetString(PyExc_RuntimeError, "Error performing inplace unary operation");
    }

    return NULL;
}

static PyObject *PyTensorBase_nb_add(PyObject *a, PyObject *b) { return PyTensorBase_nb_binary_operation(a, b, scalar_add); }
static PyObject *PyTensorBase_nb_subtract(PyObject *a, PyObject *b) { return PyTensorBase_nb_binary_operation(a, b, scalar_sub); }
static PyObject *PyTensorBase_nb_multiply(PyObject *a, PyObject *b) { return PyTensorBase_nb_binary_operation(a, b, scalar_mult); }
static PyObject *PyTensorBase_nb_floor_divide(PyObject *a, PyObject *b) { return PyTensorBase_nb_binary_operation(a, b, scalar_floordiv); }
static PyObject *PyTensorBase_nb_true_divide(PyObject *a, PyObject *b) { return PyTensorBase_nb_binary_operation(a, b, scalar_truediv); }
static PyObject *PyTensorBase_nb_power(PyObject *a, PyObject *b, PyObject *Py_UNUSED(ignored)) { return PyTensorBase_nb_binary_operation(a, b, scalar_power); }
static PyObject *PyTensorBase_nb_negative(PyObject *a) { return PyTensorBase_nb_unary_operation(a, scalar_negative); }
static PyObject *PyTensorBase_nb_positive(PyObject *a) { return PyTensorBase_nb_unary_operation(a, scalar_positive); }
static PyObject *PyTensorBase_nb_absolute(PyObject *a) { return PyTensorBase_nb_unary_operation(a, scalar_absolute); }
static PyObject *PyTensorBase_matrix_multiply(PyObject *a, PyObject *b)
{
    // Both a and b must be of type TensorBase.
    if (!(PyTensorBase_Check(a) && PyTensorBase_Check(b)))
    {
        PyErr_SetString(PyExc_ValueError, "Invalid types for addition. Must operands must be Tensors.");
        return NULL;
    }

    PyTensorBase *result = (PyTensorBase *)PyObject_New(PyTensorBase, &PyTensorBaseType);
    if (result == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create new TensorBase object.");
        return NULL;
    }

    TensorBase *l = &(((PyTensorBase *)a)->tb);
    TensorBase *r = &(((PyTensorBase *)b)->tb);

    if (TensorBase_matrix_multiply(l, r, &(result->tb)) < 0)
    {
        PyErr_SetString(PyExc_RuntimeError, "Error in matrix multiply.");
        return NULL;
    }

    return (PyObject *)result;
}

static PyObject *PyTensorBase_abs_(PyObject *self, PyObject *Py_UNUSED(args))
{
    PyErr_SetString(PyExc_NotImplementedError, "PyTensorBase_abs_ is not implemented");
    return NULL;
}

static PyObject *PyTensorBase_abs(PyObject *self, PyObject *Py_UNUSED(args))
{
    PyErr_SetString(PyExc_NotImplementedError, "PyTensorBase_abs is not implemented");
    return NULL;
}

static PyObject *PyTensorBase_cos_(PyObject *self, PyObject *Py_UNUSED(args))
{
    PyErr_SetString(PyExc_NotImplementedError, "PyTensorBase_cos_ is not implemented");
    return NULL;
}

static PyObject *PyTensorBase_cos(PyObject *self, PyObject *Py_UNUSED(args))
{
    PyErr_SetString(PyExc_NotImplementedError, "PyTensorBase_cos is not implemented");
    return NULL;
}

static PyObject *PyTensorBase_sin_(PyObject *self, PyObject *Py_UNUSED(args))
{
    PyErr_SetString(PyExc_NotImplementedError, "PyTensorBase_sin_ is not implemented");
    return NULL;
}

static PyObject *PyTensorBase_sin(PyObject *self, PyObject *Py_UNUSED(args))
{
    PyErr_SetString(PyExc_NotImplementedError, "PyTensorBase_sin is not implemented");
    return NULL;
}

static PyObject *PyTensorBase_tan_(PyObject *self, PyObject *Py_UNUSED(args))
{
    PyErr_SetString(PyExc_NotImplementedError, "PyTensorBase_tan_ is not implemented");
    return NULL;
}

static PyObject *PyTensorBase_tan(PyObject *self, PyObject *Py_UNUSED(args))
{
    PyErr_SetString(PyExc_NotImplementedError, "PyTensorBase_tan is not implemented");
    return NULL;
}

static PyObject *PyTensorBase_tanh_(PyObject *self, PyObject *Py_UNUSED(args))
{
    PyErr_SetString(PyExc_NotImplementedError, "PyTensorBase_tanh_ is not implemented");
    return NULL;
}

static PyObject *PyTensorBase_tanh(PyObject *self, PyObject *Py_UNUSED(args))
{
    PyErr_SetString(PyExc_NotImplementedError, "PyTensorBase_tanh is not implemented");
    return NULL;
}

static PyObject *PyTensorBase_log_(PyObject *self, PyObject *Py_UNUSED(args))
{
    PyErr_SetString(PyExc_NotImplementedError, "PyTensorBase_log_ is not implemented");
    return NULL;
}

static PyObject *PyTensorBase_log(PyObject *self, PyObject *Py_UNUSED(args))
{
    PyErr_SetString(PyExc_NotImplementedError, "PyTensorBase_log is not implemented");
    return NULL;
}

static PyObject *PyTensorBase_exp_(PyObject *self, PyObject *Py_UNUSED(args))
{
    PyErr_SetString(PyExc_NotImplementedError, "PyTensorBase_exp_ is not implemented");
    return NULL;
}

static PyObject *PyTensorBase_exp(PyObject *self, PyObject *Py_UNUSED(args))
{
    PyErr_SetString(PyExc_NotImplementedError, "PyTensorBase_exp is not implemented");
    return NULL;
}

static PyObject *PyTensorBase_sigmoid_(PyObject *self, PyObject *Py_UNUSED(args))
{
    PyErr_SetString(PyExc_NotImplementedError, "PyTensorBase_sigmoid_ is not implemented");
    return NULL;
}

static PyObject *PyTensorBase_sigmoid(PyObject *self, PyObject *Py_UNUSED(args))
{
    PyErr_SetString(PyExc_NotImplementedError, "PyTensorBase_sigmoid is not implemented");
    return NULL;
}

static PyObject *PyTensorBase_zero_(PyObject *self, PyObject *Py_UNUSED(args))
{
    PyErr_SetString(PyExc_NotImplementedError, "PyTensorBase_zero_ is not implemented");
    return NULL;
}

static PyObject *PyTensorBase_reshape_(PyObject *self, PyObject *args)
{
    PyErr_SetString(PyExc_NotImplementedError, "PyTensorBase_reshape_ is not implemented");
    return NULL;
}

static PyObject *PyTensorBase_reshape(PyObject *self, PyObject *args)
{
    PyErr_SetString(PyExc_NotImplementedError, "PyTensorBase_reshape is not implemented");
    return NULL;
}

static PyObject *PyTensorBase_fill_(PyObject *self, PyObject *args)
{
    PyErr_SetString(PyExc_NotImplementedError, "PyTensorBase_fill_ is not implemented");
    return NULL;
}

static PyObject *PyTensorBase_max(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    PyErr_SetString(PyExc_NotImplementedError, "PyTensorBase_max is not implemented");
    return NULL;
}

static PyObject *PyTensorBase_min(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    PyErr_SetString(PyExc_NotImplementedError, "PyTensorBase_min is not implemented");
    return NULL;
}

static PyObject *PyTensorBase_mean(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    PyErr_SetString(PyExc_NotImplementedError, "PyTensorBase_mean is not implemented");
    return NULL;
}

static PyObject *PyTensorBase_sum(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    PyErr_SetString(PyExc_NotImplementedError, "PyTensorBase_sum is not implemented");
    return NULL;
}

static PyObject *PyTensorBase_broadcast_to(PyObject *self, PyObject *args)
{
    PyErr_SetString(PyExc_NotImplementedError, "PyTensorBase_broadcast_to is not implemented");
    return NULL;
}

static PyObject *PyTensorBase_permute(PyObject *self, PyObject *args)
{
    PyErr_SetString(PyExc_NotImplementedError, "PyTensorBase_permute is not implemented");
    return NULL;
}

static PyObject *PyTensorBase_transpose(PyObject *self, PyObject *args)
{
    PyErr_SetString(PyExc_NotImplementedError, "PyTensorBase_transpose is not implemented");
    return NULL;
}

static PyObject *PyTensorBase_get_dim(PyTensorBase *self, PyObject *Py_UNUSED(ignored))
{
    return PyLong_FromLong(self->tb.ndim);
}

static PyObject *PyTensorBase_get_size(PyTensorBase *self, PyObject *Py_UNUSED(ignored))
{
    PyObject *shape = PyTuple_New(self->tb.ndim);

    for (long i = 0; i < self->tb.ndim; i++)
    {
        if (PyTuple_SetItem(shape, i, PyLong_FromLong(self->tb.shape[i])))
        {
            PyErr_SetString(PyExc_RuntimeError, "Failed to set shape item.");
            return NULL;
        }
    }

    return shape;
}

static PyObject *PyTensorBase_get_numel(PyTensorBase *self, PyObject *Py_UNUSED(ignored))
{
    return PyLong_FromLong(self->tb.numel);
}

static PyObject *PyTensorBase_get_stride(PyTensorBase *self, PyObject *Py_UNUSED(ignored))
{
    PyObject *stride = PyTuple_New(self->tb.ndim);

    for (long i = 0; i < self->tb.ndim; i++)
    {
        if (PyTuple_SetItem(stride, i, PyLong_FromLong(self->tb.stride[i])))
        {
            PyErr_SetString(PyExc_RuntimeError, "Failed to set stride item.");
            return NULL;
        }
    }

    return stride;
}

static PyObject *PyTensorBase_ones(PyModuleDef *module, PyObject *args)
{
    PyErr_SetString(PyExc_NotImplementedError, "PyTensorBase_min is not implemented");
    return NULL;
}

static PyObject *PyTensorBase_randn(PyModuleDef *module, PyObject *args)
{
    PyErr_SetString(PyExc_NotImplementedError, "PyTensorBase_min is not implemented");
    return NULL;
}

static long args_to_shape(PyObject *args, ShapeArray *tb_shape)
{
    // Parse args as tuple of dimensions (or tuple of tuple of dimensions)
    Py_ssize_t tuple_len = PyTuple_Size(args);

    if (tuple_len > MAX_RANK)
    {
        PyErr_SetString(PyExc_ValueError, "Tensor rank exceeds maximum allowed.");
        return -1;
    }

    memset(tb_shape, -1, sizeof(ShapeArray));

    for (long i = 0; i < tuple_len; i++)
    {
        PyObject *item = PyTuple_GetItem(args, i);
        if (!PyLong_Check(item))
        {
            PyErr_SetString(PyExc_ValueError, "Tensor dimensions must be integers.");
            return -1;
        }

        *(tb_shape + i) = PyLong_AsLong(item);
    }
    // return the number of dimensions
    return (long)tuple_len;
}

static int PyTensorBase_init(PyTensorBase *self, PyObject *args, PyObject *kwds)
{
    ShapeArray tb_shape = {0};
    long ndim = args_to_shape(args, &tb_shape);
    if (ndim < 0)
    {
        return -1;
    }

    if (TensorBase_init(&self->tb, tb_shape, ndim) < 0)
    {
        PyErr_SetString(PyExc_RuntimeError, "Failed to initialize tensor data.");
        return -1;
    }

    return 0;
}

static void PyTensorBase_dealloc(PyTensorBase *self)
{
    TensorBase_dealloc(&self->tb);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *PyTensorBase_str(PyTensorBase *obj)
{
    // TODO: calculate a reasonable buffer size
    char *str_buffer = malloc(100 * sizeof(char));

    TensorBase_to_string(&obj->tb, str_buffer, 100 * sizeof(char));

    return Py_BuildValue("s", str_buffer);
}
