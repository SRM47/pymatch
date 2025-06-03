#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include <stddef.h>

#include "tensorbase.h"

#define RETURN_IF_PY_ERROR \
    if (PyErr_Occurred())  \
    {                      \
        return NULL;       \
    }

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
    .nb_negative = (unaryfunc)PyTensorBase_nb_negative,
    // .nb_positive = 0,
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
 *                   Rich Comparison                     *
 *********************************************************/

static PyObject *PyTensorBase_richcompare(PyObject *self, PyObject *other, int op);

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

// https://docs.python.org/3/c-api/object.html#c.PyObject_Size
static Py_ssize_t PyTensorBase_length(PyObject *o);

// https://docs.python.org/3/c-api/object.html#c.PyObject_GetItem
static PyObject *PyTensorBase_getitem(PyObject *o, PyObject *key);

// https://docs.python.org/3/c-api/object.html#c.PyObject_SetItem
static int PyTensorBase_setitem(PyObject *o, PyObject *key, PyObject *v);

// https://docs.python.org/3/c-api/typeobj.html#mapping-object-structures
static PyMappingMethods PyTensorBase_mapping_methods = {
    .mp_length = (lenfunc)PyTensorBase_length,               // __len__
    .mp_subscript = (binaryfunc)PyTensorBase_getitem,        // __getitem__
    .mp_ass_subscript = (objobjargproc)PyTensorBase_setitem, // __setitem__
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

static PyObject *PyTensorBase_relu_(PyObject *self, PyObject *Py_UNUSED(args));
static PyObject *PyTensorBase_relu(PyObject *self, PyObject *Py_UNUSED(args));

static PyObject *PyTensorBase_zero_(PyObject *self, PyObject *Py_UNUSED(args));

static PyObject *PyTensorBase_item(PyObject *self, PyObject *Py_UNUSED(args));

static PyObject *PyTensorBase_reshape_(PyObject *self, PyObject *args);
static PyObject *PyTensorBase_reshape(PyObject *self, PyObject *args);

static PyObject *PyTensorBase_fill_(PyObject *self, PyObject *args);

static PyObject *PyTensorBase_randn_(PyObject *self, PyObject *const *args, Py_ssize_t nargs);

static PyObject *PyTensorBase_max(PyObject *self, PyObject *const *args, Py_ssize_t nargs);
static PyObject *PyTensorBase_min(PyObject *self, PyObject *const *args, Py_ssize_t nargs);
static PyObject *PyTensorBase_mean(PyObject *self, PyObject *const *args, Py_ssize_t nargs);
static PyObject *PyTensorBase_sum(PyObject *self, PyObject *const *args, Py_ssize_t nargs);

static PyObject *PyTensorBase_argmax(PyObject *self, PyObject *const *args, Py_ssize_t nargs);
static PyObject *PyTensorBase_argmin(PyObject *self, PyObject *const *args, Py_ssize_t nargs);

static PyObject *PyTensorBase_unbroadcast(PyObject *self, PyObject *args);
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

    {"relu_", (PyCFunction)PyTensorBase_relu_, METH_NOARGS, "In-place relu."},
    {"relu", (PyCFunction)PyTensorBase_relu, METH_NOARGS, "Out-of-place relu."},

    {"zero_", (PyCFunction)PyTensorBase_zero_, METH_NOARGS, "In-place zero."},

    {"item", (PyCFunction)PyTensorBase_item, METH_NOARGS, "Get the single element the array."},

    {"transpose", (PyCFunction)PyTensorBase_transpose, METH_NOARGS, "Transpose the tensor."},

    // Methods with arguments.
    {"reshape_", (PyCFunction)PyTensorBase_reshape_, METH_O, "In-place reshape."},
    {"reshape", (PyCFunction)PyTensorBase_reshape, METH_O, "Out-of-place reshape."},

    {"fill_", (PyCFunction)PyTensorBase_fill_, METH_O, "In-place fill."},

    {"randn_", (PyCFunctionFast)PyTensorBase_randn_, METH_FASTCALL, "In-place randn."},

    {"max", (PyCFunctionFast)PyTensorBase_max, METH_FASTCALL, "Compute the maximum value."},
    {"min", (PyCFunctionFast)PyTensorBase_min, METH_FASTCALL, "Compute the minimum value."},

    {"mean", (PyCFunctionFast)PyTensorBase_mean, METH_FASTCALL, "Compute the mean value."},
    {"sum", (PyCFunctionFast)PyTensorBase_sum, METH_FASTCALL, "Compute the sum of elements."},

    {"argmax", (PyCFunctionFast)PyTensorBase_argmax, METH_FASTCALL, "Compute the indices of the maximum value of all elements in the input tensor."},
    {"argmin", (PyCFunctionFast)PyTensorBase_argmin, METH_FASTCALL, "Compute the indices of the minimum value of all elements in the input tensor."},

    {"unbroadcast", (PyCFunction)PyTensorBase_unbroadcast, METH_O, "Unbroadcast TensorBase."},

    {"permute", (PyCFunction)PyTensorBase_permute, METH_O, "Permute the dimensions of the array."},
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
static PyObject *PyTensorBase_get_raw_data(PyTensorBase *self, PyObject *Py_UNUSED(ignored));

static PyGetSetDef PyTensorBase_getset[] = {
    {"dim", (getter)PyTensorBase_get_dim, NULL, "Gets tensor rank", NULL},
    {"size", (getter)PyTensorBase_get_size, NULL, "Tensor shape", NULL},
    {"numel", (getter)PyTensorBase_get_numel, NULL, "Number of elements in Tensor", NULL},
    {"stride", (getter)PyTensorBase_get_stride, NULL, "Strides of tensor", NULL},
    {"_raw_data", (getter)PyTensorBase_get_raw_data, NULL, "The raw data of the tensorbase.", NULL},
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
    PyVarObject_HEAD_INIT(NULL, 0)
        .tp_name = "tensorbase.TensorBase", /* For printing, in format "<module>.<name>" */
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
    .tp_str = (reprfunc)PyTensorBase_str,
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
    .tp_richcompare = (richcmpfunc)PyTensorBase_richcompare,

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
    .m_name = "tensorbase",
    .m_doc = PyDoc_STR("TODO: docs"),
    .m_size = -1,
    .m_methods = 0,
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

static scalar PyFloatOrLong_asDouble(PyObject *obj)
{
    if (PyLong_Check(obj))
    {
        return PyLong_AsDouble(obj);
    }
    return PyFloat_AsDouble(obj);
}
static long arg_to_shape(PyObject *arg, ShapeArray tb_shape)
{
    PyObject *shape_array = arg;

    // Parse args as tuple of dimensions (or tuple of tuple of dimensions)
    Py_ssize_t tuple_len = PyTuple_Size(shape_array);

    if (tuple_len > MAX_RANK)
    {
        PyErr_SetString(PyExc_ValueError, "Provided shape exceeds maximum allowed rank.");
        return -1;
    }

    for (long i = 0; i < MAX_RANK; i++)
    {
        tb_shape[i] = -1.0;
    }

    for (long i = 0; i < tuple_len; i++)
    {
        PyObject *item = PyTuple_GetItem(shape_array, i);
        if (item == NULL)
        {
            PyErr_SetString(PyExc_ValueError, "Failed to retrieve an item from the tuple.");
            return -1;
        }

        if (!PyLong_Check(item))
        {
            PyErr_SetString(PyExc_ValueError, "Tensor values must be integers!!");
            return -1;
        }

        tb_shape[i] = PyLong_AsLong(item);
        if (tb_shape[i] == -1 && PyErr_Occurred())
        {
            return -1;
        }
    }
    // return the number of dimensions
    return (long)tuple_len;
}

static long args_to_shape(PyObject *args, ShapeArray tb_shape)
{
    if (!(PyTuple_Size(args) == 1 && PyTuple_Check(PyTuple_GetItem(args, 0))))
    {
        PyErr_SetString(PyExc_ValueError, "Expected tuple.");
        return -1;
    }

    return arg_to_shape(PyTuple_GetItem(args, 0), tb_shape);
}

static PyObject *PyTensorBase_nb_binary_operation(PyObject *a, PyObject *b, BinaryScalarOperation binop)
{
    StatusCode status = OK;
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
        status = TensorBase_binary_op_tensorbase_scalar(t, s, &(result->tb), binop);
    }
    // (Long | Float) + PyTensorBase
    else if (PyFloatOrLong_Check(a) && PyTensorBase_Check(b))
    {
        TensorBase *t = &(((PyTensorBase *)b)->tb);
        scalar s = PyFloatOrLong_asDouble(a);
        status = TensorBase_binary_op_scalar_tensorbase(t, s, &(result->tb), binop);
    }
    // PyTensorBase + PyTensorBase
    else if (PyTensorBase_Check(a) && PyTensorBase_Check(b))
    {
        TensorBase *l = &(((PyTensorBase *)a)->tb);
        TensorBase *r = &(((PyTensorBase *)b)->tb);
        status = TensorBase_binary_op_tensorbase_tensorbase(l, r, &(result->tb), binop);
    }
    // Incompatible types for mathematical binary operations
    else
    {
        PyErr_SetString(PyExc_ValueError, "Invalid types for addition. Either argument must be Tensor (non-singleton), Float, or Integer.");
    }

    // Check Error Codes
    switch (status)
    {
    case OK:
        break;
    case NULL_INPUT_ERR:
        PyErr_SetString(PyExc_RuntimeError, "Null pointer provided to binary operation");
        return NULL;
    case MALLOC_ERR:
        PyErr_SetString(PyExc_RuntimeError, "Unable to allocate memory for new tensorbase object.");
        return NULL;
    case INCOMPATABLE_BROASCAST_SHAPES:
        PyErr_SetString(PyExc_RuntimeError, "Incompatable shapes to broadcast for binary operation.");
        return NULL;
    default:
        PyErr_SetString(PyExc_RuntimeError, "Unknown Error Occured.");
        return NULL;
    }

    return (PyObject *)result;
}

static PyObject *PyTensorBase_nb_unary_operation(PyObject *a, UnaryScalarOperation uop)
{
    PyTensorBase *result = (PyTensorBase *)PyObject_New(PyTensorBase, &PyTensorBaseType);
    if (result == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create new TensorBase object.");
        return NULL;
    }

    TensorBase *in = &(((PyTensorBase *)a)->tb);
    StatusCode status = TensorBase_unary_op(in, &(result->tb), uop);

    // Check Error Codes
    switch (status)
    {
    case OK:
        break;
    case NULL_INPUT_ERR:
        PyErr_SetString(PyExc_RuntimeError, "Null pointer provided to binary operation");
        return NULL;
    case MALLOC_ERR:
        PyErr_SetString(PyExc_RuntimeError, "Unable to allocate memory for new tensorbase object.");
        return NULL;
    default:
        PyErr_SetString(PyExc_RuntimeError, "Unknown Error Occured.");
        return NULL;
    }

    return (PyObject *)result;
}

static PyObject *PyTensorBase_nb_unary_operation_inplace(PyObject *a, UnaryScalarOperation uop)
{
    // Assumes input PyObject is already of type PyTensorBase.
    TensorBase *in = &(((PyTensorBase *)a)->tb);
    StatusCode status = TensorBase_unary_op_inplace(in, uop);
    switch (status)
    {
    case OK:
        break;
    case NULL_INPUT_ERR:
        PyErr_SetString(PyExc_RuntimeError, "Null pointer provided to binary operation");
        return NULL;
    case MALLOC_ERR:
        PyErr_SetString(PyExc_RuntimeError, "Unable to allocate memory for new tensorbase object.");
        return NULL;
    default:
        PyErr_SetString(PyExc_RuntimeError, "Unknown Error Occured.");
        return NULL;
    }

    Py_RETURN_NONE;
}

static PyObject *PyTensorBase_nb_add(PyObject *a, PyObject *b) { return PyTensorBase_nb_binary_operation(a, b, SCALAR_ADD); }
static PyObject *PyTensorBase_nb_subtract(PyObject *a, PyObject *b) { return PyTensorBase_nb_binary_operation(a, b, SCALAR_SUB); }
static PyObject *PyTensorBase_nb_multiply(PyObject *a, PyObject *b) { return PyTensorBase_nb_binary_operation(a, b, SCALAR_MULT); }
static PyObject *PyTensorBase_nb_floor_divide(PyObject *a, PyObject *b) { return PyTensorBase_nb_binary_operation(a, b, SCALAR_FLOORDIV); }
static PyObject *PyTensorBase_nb_true_divide(PyObject *a, PyObject *b) { return PyTensorBase_nb_binary_operation(a, b, SCALAR_TRUEDIV); }
static PyObject *PyTensorBase_nb_power(PyObject *a, PyObject *b, PyObject *Py_UNUSED(ignored)) { return PyTensorBase_nb_binary_operation(a, b, SCALAR_POWER); }
static PyObject *PyTensorBase_nb_negative(PyObject *a) { return PyTensorBase_nb_unary_operation(a, SCALAR_NEGATIVE); }
static PyObject *PyTensorBase_nb_absolute(PyObject *a) { return PyTensorBase_nb_unary_operation(a, SCALAR_ABSOLUTE); }
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

    StatusCode status = TensorBase_matrix_multiply(l, r, &(result->tb));
    switch (status)
    {
    case OK:
        break;
    case NULL_INPUT_ERR:
        PyErr_SetString(PyExc_RuntimeError, "Null tensorbase objects provided to matmul method.");
        return NULL;
    case MATMUL_SINGLETON:
        PyErr_SetString(PyExc_RuntimeError, "Matrix multiplication is not supported for singleton (ndim = 0) operands. Both operands must be at least 1 dimensional.");
        return NULL;
    case MATMUL_INCOMPATABLE_SHAPES:
        PyErr_SetString(PyExc_RuntimeError, "Incompatable shapes for matrix multiplication.");
        return NULL;
    case INCOMPATABLE_BROASCAST_SHAPES:
        PyErr_SetString(PyExc_RuntimeError, "Unbroadcastable shapes for matrix multiplication.");
        return NULL;
    default:
        PyErr_SetString(PyExc_RuntimeError, "Unknown error occured in matrix multiplication.");
        return NULL;
    }

    return (PyObject *)result;
}

static PyObject *PyTensorBase_relu_(PyObject *self, PyObject *Py_UNUSED(args)) { return PyTensorBase_nb_unary_operation_inplace(self, SCALAR_RELU); }
static PyObject *PyTensorBase_relu(PyObject *self, PyObject *Py_UNUSED(args)) { return PyTensorBase_nb_unary_operation(self, SCALAR_RELU); }

static PyObject *PyTensorBase_abs_(PyObject *self, PyObject *Py_UNUSED(args)) { return PyTensorBase_nb_unary_operation_inplace(self, SCALAR_ABSOLUTE); }
static PyObject *PyTensorBase_abs(PyObject *self, PyObject *Py_UNUSED(args)) { return PyTensorBase_nb_unary_operation(self, SCALAR_ABSOLUTE); }

static PyObject *PyTensorBase_cos_(PyObject *self, PyObject *Py_UNUSED(args))
{
    return PyTensorBase_nb_unary_operation_inplace(self, SCALAR_COS);
}

static PyObject *PyTensorBase_cos(PyObject *self, PyObject *Py_UNUSED(args))
{
    return PyTensorBase_nb_unary_operation(self, SCALAR_COS);
}

static PyObject *PyTensorBase_sin_(PyObject *self, PyObject *Py_UNUSED(args))
{
    return PyTensorBase_nb_unary_operation_inplace(self, SCALAR_SIN);
}

static PyObject *PyTensorBase_sin(PyObject *self, PyObject *Py_UNUSED(args))
{
    return PyTensorBase_nb_unary_operation(self, SCALAR_SIN);
}

static PyObject *PyTensorBase_tan_(PyObject *self, PyObject *Py_UNUSED(args))
{
    return PyTensorBase_nb_unary_operation_inplace(self, SCALAR_TAN);
}

static PyObject *PyTensorBase_tan(PyObject *self, PyObject *Py_UNUSED(args))
{
    return PyTensorBase_nb_unary_operation(self, SCALAR_TAN);
}

static PyObject *PyTensorBase_tanh_(PyObject *self, PyObject *Py_UNUSED(args))
{
    return PyTensorBase_nb_unary_operation_inplace(self, SCALAR_TANH);
}

static PyObject *PyTensorBase_tanh(PyObject *self, PyObject *Py_UNUSED(args))
{
    return PyTensorBase_nb_unary_operation(self, SCALAR_TANH);
}

static PyObject *PyTensorBase_log_(PyObject *self, PyObject *Py_UNUSED(args))
{
    return PyTensorBase_nb_unary_operation_inplace(self, SCALAR_LOG);
}

static PyObject *PyTensorBase_log(PyObject *self, PyObject *Py_UNUSED(args))
{
    return PyTensorBase_nb_unary_operation(self, SCALAR_LOG);
}

static PyObject *PyTensorBase_exp_(PyObject *self, PyObject *Py_UNUSED(args))
{
    return PyTensorBase_nb_unary_operation_inplace(self, SCALAR_EXP);
}

static PyObject *PyTensorBase_exp(PyObject *self, PyObject *Py_UNUSED(args))
{
    return PyTensorBase_nb_unary_operation(self, SCALAR_EXP);
}

static PyObject *PyTensorBase_sigmoid_(PyObject *self, PyObject *Py_UNUSED(args))
{
    return PyTensorBase_nb_unary_operation_inplace(self, SCALAR_SIGMOID);
}

static PyObject *PyTensorBase_sigmoid(PyObject *self, PyObject *Py_UNUSED(args))
{
    return PyTensorBase_nb_unary_operation(self, SCALAR_SIGMOID);
}

static PyObject *PyTensorBase_zero_(PyObject *self, PyObject *Py_UNUSED(args))
{
    PyErr_SetString(PyExc_NotImplementedError, "PyTensorBase_zero_ is not implemented");
    return NULL;
}

static PyObject *PyTensorBase_item(PyObject *self, PyObject *Py_UNUSED(args))
{
    scalar item;
    TensorBase *t = &((PyTensorBase *)self)->tb;
    StatusCode status = TensorBase_item(t, &item);
    if (status == ITEM_NUMEL_NOT_ONE)
    {
        PyErr_SetString(PyExc_NotImplementedError, "item() is supported only for tensors with only 1 element (numel = 1).");
        return NULL;
    }

    return PyFloat_FromDouble(item);
}

static PyObject *PyTensorBase_reshape_(PyObject *self, PyObject *args)
{
    TensorBase *t = &((PyTensorBase *)self)->tb;
    ShapeArray shape;
    long ndim = arg_to_shape(args, shape);
    if (ndim < 0)
    {
        return NULL;
    }

    StatusCode status = TensorBase_reshape_inplace(t, shape, ndim);
    switch (status)
    {
    case OK:
        break;
    case NULL_INPUT_ERR:
        PyErr_SetString(PyExc_RuntimeError, "Null tensorbase objects provided to reshape_.");
        return NULL;
    case NDIM_OUT_OF_BOUNDS:
        PyErr_SetString(PyExc_RuntimeError, "Maximum tensor rank exceeded.");
        return NULL;
    case INVALID_DIMENSION_SIZE:
        PyErr_SetString(PyExc_RuntimeError, "All dimensions in tensor shape must be non negative.");
        return NULL;
    case RESHAPE_INVALID_SHAPE_NUMEL_MISMATCH:
        PyErr_SetString(PyExc_RuntimeError, "Unable to reshape because number different of elements in new tensor.");
        return NULL;
    case MALLOC_ERR:
        PyErr_SetString(PyExc_RuntimeError, "Memory allocation error, unable to allocate enough memory for new tensorbase object.");
        return NULL;
    default:
        PyErr_SetString(PyExc_RuntimeError, "Unknown Error.");
        return NULL;
    }

    Py_RETURN_NONE;
}

static PyObject *PyTensorBase_reshape(PyObject *self, PyObject *args)
{
    PyTensorBase *result = (PyTensorBase *)PyObject_New(PyTensorBase, &PyTensorBaseType);

    TensorBase *in = &((PyTensorBase *)self)->tb;
    TensorBase *out = &((PyTensorBase *)result)->tb;

    ShapeArray shape;
    long ndim = arg_to_shape(args, shape);
    if (ndim < 0)
    {
        return NULL;
    }

    StatusCode status = TensorBase_reshape(in, out, shape, ndim);
    switch (status)
    {
    case OK:
        break;
    case NULL_INPUT_ERR:
        PyErr_SetString(PyExc_RuntimeError, "Null tensorbase objects provided to reshape_.");
        return NULL;
    case NDIM_OUT_OF_BOUNDS:
        PyErr_SetString(PyExc_RuntimeError, "Maximum tensor rank exceeded.");
        return NULL;
    case INVALID_DIMENSION_SIZE:
        PyErr_SetString(PyExc_RuntimeError, "All dimensions in tensor shape must be non negative.");
        return NULL;
    case RESHAPE_INVALID_SHAPE_NUMEL_MISMATCH:
        PyErr_SetString(PyExc_RuntimeError, "Unable to reshape because number different of elements in new tensor.");
        return NULL;
    case MALLOC_ERR:
        PyErr_SetString(PyExc_RuntimeError, "Memory allocation error, unable to allocate enough memory for new tensorbase object.");
        return NULL;
    default:
        PyErr_SetString(PyExc_RuntimeError, "Unknown Error.");
        return NULL;
    }

    return (PyObject *)result;
}

static PyObject *PyTensorBase_fill_(PyObject *self, PyObject *args)
{
    if (!PyFloatOrLong_Check(args))
    {
        PyErr_SetString(PyExc_RuntimeError, "Operand must be a Float or Integer.");
        return NULL;
    }
    scalar fill_value = PyFloatOrLong_asDouble(args);

    if (PyErr_Occurred())
    {
        PyErr_SetString(PyExc_RuntimeError, "Unable to parse provided scalar.");
        return NULL;
    }

    TensorBase *t = &((PyTensorBase *)self)->tb;
    StatusCode status = TensorBase_fill_(t, fill_value);
    switch (status)
    {
    case OK:
        break;
    case NULL_INPUT_ERR:
        PyErr_SetString(PyExc_RuntimeError, "Null tensorbase objects provided to fill_.");
        return NULL;
    default:
        PyErr_SetString(PyExc_RuntimeError, "Unknown Error in fill_.");
        return NULL;
    }

    Py_RETURN_NONE;
}

static PyObject *PyTensorBase_agg(PyObject *self, PyObject *const *args, Py_ssize_t nargs, AggScalarOperation agg)
{
    if (nargs != 2)
    {
        PyErr_SetString(PyExc_RuntimeError, "Expect 2 arguments");
        return NULL;
    }

    // Both a and b must be of type TensorBase.
    if (!(PyTuple_Check(args[0]) && PyBool_Check(args[1])))
    {
        PyErr_SetString(PyExc_ValueError, "Invalid types for sum. Expected tuple, bool");
        return NULL;
    }

    TensorBase *t = &(((PyTensorBase *)self)->tb);

    IndexArray dims;
    if (PyTuple_Size(args[0]) != 0)
    {
        // If specific dimensions were provided, aggregate over the specified dimensions.
        // Covert the tuple of dimensions into an IndexArray.
        if (arg_to_shape(args[0], dims) < 0)
        {
            return NULL;
        }
    }
    else
    {
        // If dims = None, then aggregate over all dimensions, synonymous with aggregating over dimensions 1, 2,..., ndim-1.
        for (long i = 0; i < t->ndim; i++)
        {
            dims[i] = i;
        }
        for (long i = t->ndim; i < MAX_RANK; i++)
        {
            dims[i] = -1;
        }
    }
    int keepdim = PyObject_IsTrue(args[1]);
    if (keepdim == -1)
    {
        PyErr_SetString(PyExc_ValueError, "Invalid argument types for sum(). Expected tuple[int], bool");
        return NULL;
    }

    PyTensorBase *result = (PyTensorBase *)PyObject_New(PyTensorBase, &PyTensorBaseType);
    if (result == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create new TensorBase object.");
        return NULL;
    }

    StatusCode status = TensorBase_aggregate(t, dims, keepdim, &(result->tb), agg);
    switch (status)
    {
    case OK:
        break;
    case NULL_INPUT_ERR:
        PyErr_SetString(PyExc_RuntimeError, "Null tensorbase objects provided to reshape_.");
        return NULL;
    case NDIM_OUT_OF_BOUNDS:
        PyErr_SetString(PyExc_RuntimeError, "Maximum tensor rank exceeded.");
        return NULL;
    case INVALID_DIMENSION_SIZE:
        PyErr_SetString(PyExc_RuntimeError, "All dimensions in tensor shape must be non negative.");
        return NULL;
    case RESHAPE_INVALID_SHAPE_NUMEL_MISMATCH:
        PyErr_SetString(PyExc_RuntimeError, "Unable to reshape because number different of elements in new tensor.");
        return NULL;
    case MALLOC_ERR:
        PyErr_SetString(PyExc_RuntimeError, "Memory allocation error, unable to allocate enough memory for new tensorbase object.");
        return NULL;
    case DUPLICATE_AGGREGATION_DIM:
        PyErr_SetString(PyExc_RuntimeError, "Duplicate dimension provided in dims to aggregate over.");
        return NULL;
    default:
        PyErr_SetString(PyExc_RuntimeError, "Unknown Error.");
        return NULL;
    }

    return (PyObject *)result;
}

static PyObject *PyTensorBase_max(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    // The first argument must be a tuple containing at most one element.
    if (!(PyTuple_Check(args[0]) && (PyTuple_Size(args[0]) <= 1)))
    {
        PyErr_SetString(PyExc_ValueError, "Expected max over a single, or all dimensions (dim = None)");
        return NULL;
    }
    return PyTensorBase_agg(self, args, nargs, SCALAR_AGG_MAX);
}

static PyObject *PyTensorBase_min(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    // The first argument must be a tuple containing at most one element.
    if (!(PyTuple_Check(args[0]) && (PyTuple_Size(args[0]) <= 1)))
    {
        PyErr_SetString(PyExc_ValueError, "Expected min over a single, or all dimensions (dim = None)");
        return NULL;
    }
    return PyTensorBase_agg(self, args, nargs, SCALAR_AGG_MIN);
}

static PyObject *PyTensorBase_argmax(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    // The first argument must be a tuple containing at most one element.
    if (!(PyTuple_Check(args[0]) && (PyTuple_Size(args[0]) <= 1)))
    {
        PyErr_SetString(PyExc_ValueError, "Expected argmax over a single, or all dimensions (dim = None)");
        return NULL;
    }
    return PyTensorBase_agg(self, args, nargs, SCALAR_AGG_ARGMAX);
}

static PyObject *PyTensorBase_argmin(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    // The first argument must be a tuple containing at most one element.
    if (!(PyTuple_Check(args[0]) && (PyTuple_Size(args[0]) <= 1)))
    {
        PyErr_SetString(PyExc_ValueError, "Expected argmin over a single, or all dimensions (dim = None)");
        return NULL;
    }
    return PyTensorBase_agg(self, args, nargs, SCALAR_AGG_ARGMIN);
}

static PyObject *PyTensorBase_mean(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    return PyTensorBase_agg(self, args, nargs, SCALAR_AGG_MEAN);
}

static PyObject *PyTensorBase_sum(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    return PyTensorBase_agg(self, args, nargs, SCALAR_AGG_SUM);
}

static PyObject *PyTensorBase_broadcast_to(PyObject *self, PyObject *args)
{
    PyErr_SetString(PyExc_NotImplementedError, "PyTensorBase_broadcast_to is not implemented");
    return NULL;
}

static PyObject *PyTensorBase_permute(PyObject *self, PyObject *args)
{
    PyTensorBase *result = (PyTensorBase *)PyObject_New(PyTensorBase, &PyTensorBaseType);

    TensorBase *in = &((PyTensorBase *)self)->tb;
    TensorBase *out = &((PyTensorBase *)result)->tb;

    IndexArray permutation;
    long ndim = arg_to_shape(args, permutation);
    if (ndim < 0)
    {
        return NULL;
    }

    StatusCode status = TensorBase_permute(in, permutation, ndim, out);
    switch (status)
    {
    case OK:
        break;
    case NULL_INPUT_ERR:
        PyErr_SetString(PyExc_RuntimeError, "Null tensorbase objects provided to permute.");
        return NULL;
    case NDIM_OUT_OF_BOUNDS:
        PyErr_SetString(PyExc_RuntimeError, "Maximum tensor rank exceeded.");
        return NULL;
    case MALLOC_ERR:
        PyErr_SetString(PyExc_RuntimeError, "Memory allocation error, unable to allocate enough memory for new tensorbase object.");
        return NULL;
    case PERMUTATION_DUPLICATE_DIM:
    case PERMUTATION_INCORRECT_NDIM:
    case INVALID_DIMENSION:
        PyErr_SetString(PyExc_RuntimeError, "Invalid permutation provided. Must be a valid permutation of numbers from 0 to ndim-1.");
        return NULL;
    case INVALID_DIMENSION_SIZE:
        PyErr_SetString(PyExc_RuntimeError, "All dimensions in tensor shape must be non negative.");
        return NULL;
    default:
        PyErr_SetString(PyExc_RuntimeError, "Unknown Error.");
        return NULL;
    }

    return (PyObject *)result;
}

static PyObject *PyTensorBase_transpose(PyObject *self, PyObject *Py_UNUSED(args))
{
    PyTensorBase *result = (PyTensorBase *)PyObject_New(PyTensorBase, &PyTensorBaseType);

    TensorBase *in = &(((PyTensorBase *)self)->tb);
    TensorBase *out = &(((PyTensorBase *)result)->tb);

    if (result == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create new TensorBase object.");
        return NULL;
    }

    StatusCode status = TensorBase_transpose(in, out);
    switch (status)
    {
    case OK:
        break;
    case NULL_INPUT_ERR:
        PyErr_SetString(PyExc_RuntimeError, "Null tensorbase objects provided to reshape_.");
        return NULL;
    case NDIM_OUT_OF_BOUNDS:
        PyErr_SetString(PyExc_RuntimeError, "Maximum tensor rank exceeded.");
        return NULL;
    case INVALID_DIMENSION_SIZE:
        PyErr_SetString(PyExc_RuntimeError, "All dimensions in tensor shape must be non negative.");
        return NULL;
    case RESHAPE_INVALID_SHAPE_NUMEL_MISMATCH:
        PyErr_SetString(PyExc_RuntimeError, "Unable to reshape because number different of elements in new tensor.");
        return NULL;
    case MALLOC_ERR:
        PyErr_SetString(PyExc_RuntimeError, "Memory allocation error, unable to allocate enough memory for new tensorbase object.");
        return NULL;
    case DUPLICATE_AGGREGATION_DIM:
        PyErr_SetString(PyExc_RuntimeError, "Duplicate dimension provided in dims to aggregate over.");
        return NULL;
    default:
        PyErr_SetString(PyExc_RuntimeError, "Unknown Error.");
        return NULL;
    }

    return (PyObject *)result;
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
        if (PyTuple_SetItem(stride, i, PyLong_FromLong(self->tb.strides[i])))
        {
            PyErr_SetString(PyExc_RuntimeError, "Failed to set stride item.");
            return NULL;
        }
    }

    return stride;
}

static PyObject *PyTensorBase_get_raw_data(PyTensorBase *self, PyObject *Py_UNUSED(ignored))
{
    if (self->tb.ndim == 0)
    {
        scalar value;
        memcpy(&value, &self->tb.data, sizeof(scalar));
        return PyFloat_FromDouble(value);
    }

    PyObject *raw_data = PyList_New(self->tb.numel);

    for (long i = 0; i < self->tb.numel; i++)
    {
        if (PyList_SetItem(raw_data, i, PyFloat_FromDouble(self->tb.data[i])))
        {
            PyErr_SetString(PyExc_RuntimeError, "Failed to set stride item.");
            return NULL;
        }
    }

    return raw_data;
}

static PyObject *PyTensorBase_randn_(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 2)
    {
        PyErr_SetString(PyExc_RuntimeError, "Must be exactly 2 arguments, expected Float|Int, Float|Int.");
        return NULL;
    }
    if (!(PyFloatOrLong_Check(args[0]) && PyFloatOrLong_Check(args[1])))
    {
        PyErr_SetString(PyExc_RuntimeError, "Must be exactly 2 arguments, expected Float|Int, Float|Int.");
        return NULL;
    }

    scalar mu = PyLong_AsDouble(args[0]);
    scalar sigma = PyLong_AsDouble(args[1]);

    if (PyErr_Occurred())
    {
        PyErr_SetString(PyExc_RuntimeError, "Error is parsing arguments in randn.");
        return NULL;
    }

    TensorBase *t = &((PyTensorBase *)self)->tb;
    StatusCode status = TensorBase_randn_(t, mu, sigma);

    switch (status)
    {
    case OK:
        break;
    case NULL_INPUT_ERR:
        PyErr_SetString(PyExc_RuntimeError, "Null tensorbase objects provided to randn.");
        return NULL;
    default:
        PyErr_SetString(PyExc_RuntimeError, "Unknown Error.");
        return NULL;
    }

    Py_RETURN_NONE;
}

static int PyTensorBase_init(PyTensorBase *self, PyObject *args, PyObject *kwds)
{
    if (kwds && PyDict_Size(kwds) > 0)
    {
        PyErr_SetString(PyExc_TypeError, "Tensor initialization does not accept keyword arguments.");
        return -1;
    }

    ShapeArray tb_shape;
    long ndim = args_to_shape(args, tb_shape);
    if (ndim < 0)
    {
        return -1;
    }

    // Initialize the tensor using TensorBase_init
    StatusCode status = TensorBase_init(&self->tb, tb_shape, ndim);
    switch (status)
    {
    case OK:
        break;
    case NULL_INPUT_ERR:
        PyErr_SetString(PyExc_RuntimeError, "Null tensorbase objects provided to init method.");
        return -1;
    case NDIM_OUT_OF_BOUNDS:
        PyErr_SetString(PyExc_RuntimeError, "Maximum tensor rank exceeded.");
        return -1;
    case MALLOC_ERR:
        PyErr_SetString(PyExc_RuntimeError, "Memory allocation error, unable to allocate enough memory for new tensorbase object.");
        return -1;
    default:
        PyErr_SetString(PyExc_RuntimeError, "Unknown Error.");
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
    char *str_buffer = malloc(1 * sizeof(char));
    *str_buffer = ' ';

    // TensorBase_to_string(&obj->tb, str_buffer, 100 * sizeof(char));

    TensorBase_to_string(&obj->tb);
    return Py_BuildValue("s", str_buffer);
}

static PyObject *PyTensorBase_unbroadcast(PyObject *self, PyObject *args)
{
    PyTensorBase *result = (PyTensorBase *)PyObject_New(PyTensorBase, &PyTensorBaseType);

    TensorBase *in = &((PyTensorBase *)self)->tb;
    TensorBase *out = &((PyTensorBase *)result)->tb;

    ShapeArray shape;
    long ndim = arg_to_shape(args, shape);
    if (ndim < 0)
    {
        return NULL;
    }

    StatusCode status = TensorBase_unbroadcast(in, shape, ndim, out);
    switch (status)
    {
    case OK:
        break;
    default:
        PyErr_SetString(PyExc_RuntimeError, "Unknown Error in unbroadcast.");
        return NULL;
    }

    return (PyObject *)result;
}

static PyObject *PyTensorBase_richcompare(PyObject *self, PyObject *other, int op)
{
    BinaryScalarOperation binop;
    switch (op)
    {
    case Py_LT:
        binop = SCALAR_LT;
        break;
    case Py_LE:
        binop = SCALAR_LEQ;
        break;
    case Py_GT:
        binop = SCALAR_GT;
        break;
    case Py_GE:
        binop = SCALAR_GEQ;
        break;
    case Py_EQ:
        binop = SCALAR_EQ;
        break;
    case Py_NE:
        binop = SCALAR_NEQ;
        break;
    default:
        PyErr_SetString(PyExc_NotImplementedError, "Unsupported Operation");
        return NULL;
    }

    return PyTensorBase_nb_binary_operation(self, other, binop);
}

long parse_key_to_subscripts(PyObject *key, SubscriptArray subscripts)
{
    memset(subscripts, 0, MAX_RANK * sizeof(TensorBaseSubscript));
    // If key is not a tuple, convert it to a single-element tuple
    PyObject *tuple_key = PyTuple_Check(key) ? key : PyTuple_Pack(1, key);
    Py_ssize_t ndims = PyTuple_Size(tuple_key);

    for (Py_ssize_t i = 0; i < ndims; i++)
    {
        PyObject *item = PyTuple_GetItem(tuple_key, i);
        TensorBaseSubscript *subscript = subscripts + i;

        if (PyLong_Check(item))
        {
            long index = PyLong_AsLong(item);
            if (index < 0)
            {
                PyErr_SetString(PyExc_ValueError, "Negative indices are not supported");
                return -1;
            }

            subscript->type = INDEX;
            subscript->start = index;
            subscript->stop = 0; // Not used for INDEX type
            subscript->step = 0; // Not used for INDEX type
        }
        else if (PySlice_Check(item))
        {
            Py_ssize_t start, stop, step;
            if (PySlice_Unpack(item, &start, &stop, &step) < 0)
            {
                PyErr_SetString(PyExc_ValueError, "Unable to unpack slice object in key");
                return -1;
            }
            Py_ssize_t _ = PySlice_AdjustIndices(PY_SSIZE_T_MAX, &start, &stop, step);

            if (start < 0 || stop < 0 || step < 0)
            {
                PyErr_SetString(PyExc_ValueError, "Negative indices are not supported");
                return -1;
            }

            subscript->type = SLICE;
            subscript->start = (long)start;
            subscript->stop = (long)stop;
            subscript->step = (long)step;
        }
        // Handle invalid type
        else
        {
            PyErr_SetString(PyExc_TypeError, "Invalid index type: must be integer or slice");
            return -1;
        }
    }

    return (long)ndims;
}

static Py_ssize_t PyTensorBase_length(PyObject *o)
{
    return (Py_ssize_t)((PyTensorBase *)o)->tb.ndim;
}

static PyObject *PyTensorBase_getitem(PyObject *o, PyObject *key)
{
    SubscriptArray subscripts;
    long num_subscripts = parse_key_to_subscripts(key, subscripts);
    if (num_subscripts < 0)
    {
        // Error occured;
        return NULL;
    }

    PyTensorBase *result = (PyTensorBase *)PyObject_New(PyTensorBase, &PyTensorBaseType);

    TensorBase *in = &((PyTensorBase *)o)->tb;
    TensorBase *out = &((PyTensorBase *)result)->tb;

    StatusCode status = TensorBase_get(in, subscripts, num_subscripts, out);
    switch (status)
    {
    case OK:
        break;
    case NOT_IMPLEMENTED:
        PyErr_SetString(PyExc_NotImplementedError, "__getitem__ is not yet implemented for Tensors.");
        return NULL;
    default:
        PyErr_SetString(PyExc_RuntimeError, "Unknown Error.");
        return NULL;
    }

    return (PyObject *)result;
}

static int PyTensorBase_setitem(PyObject *o, PyObject *key, PyObject *v)
{
    SubscriptArray subscripts;
    memset(subscripts, 0, MAX_RANK * sizeof(TensorBaseSubscript));
    long num_subscripts = parse_key_to_subscripts(key, subscripts);
    if (num_subscripts < 0)
    {
        // Error occured;
        return -1;
    }

    TensorBase *in = &((PyTensorBase *)o)->tb;

    if (PyFloatOrLong_Check(v))
    {
        scalar s = PyFloatOrLong_asDouble(v);
        StatusCode status = TensorBase_set_scalar(in, subscripts, num_subscripts, s);
        switch (status)
        {
        case OK:
            break;
        case NOT_IMPLEMENTED:
            PyErr_SetString(PyExc_NotImplementedError, "__setitem__ is not yet implemented for scalars.");
            return -1;
        default:
            PyErr_SetString(PyExc_RuntimeError, "Unknown Error");
            return -1;
        }
    }
    else if (PyTensorBase_Check(v))
    {
        TensorBase *t = &((PyTensorBase *)v)->tb;
        StatusCode status = TensorBase_set_tensorbase(in, subscripts, num_subscripts, t);
        switch (status)
        {
        case OK:
            break;
        case NOT_IMPLEMENTED:
            PyErr_SetString(PyExc_NotImplementedError, "__setitem__ is not yet implemented for Tensors.");
            return -1;
        default:
            PyErr_SetString(PyExc_RuntimeError, "Unknown Error");
            return -1;
        }
    }
    else
    {
        PyErr_SetString(PyExc_ValueError, "Set value must either be a Float, or another Tensorbase object.");
        return NULL;
    }

    return 0;
}
