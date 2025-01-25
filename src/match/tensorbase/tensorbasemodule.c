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

// These methods are implemented below
static PyObject *PyTensorBase_nb_ternary_operation(PyObject *a, PyObject *b, PyObject *c, scalar (*op)(scalar, scalar));
static PyObject *PyTensorBase_nb_binary_operation(PyObject *a, PyObject *b, scalar (*op)(scalar, scalar));
static PyObject *PyTensorBase_nb_unary_operation(PyObject *a, scalar (*op)(scalar, scalar));

static PyObject *PyTensorBase_matrix_multiply(PyObject *a, PyObject *b);

static PyObject *PyTensorBase_nb_add(PyObject *a, PyObject *b) { return PyTensorBase_nb_binary_operation(a, b, scalar_add); }
static PyObject *PyTensorBase_nb_subtract(PyObject *a, PyObject *b) { return PyTensorBase_nb_binary_operation(a, b, scalar_sub); }
static PyObject *PyTensorBase_nb_multiply(PyObject *a, PyObject *b) { return PyTensorBase_nb_binary_operation(a, b, scalar_mult); }
static PyObject *PyTensorBase_nb_floor_divide(PyObject *a, PyObject *b) { return PyTensorBase_nb_binary_operation(a, b, scalar_floordiv); }
static PyObject *PyTensorBase_nb_true_divide(PyObject *a, PyObject *b) { return PyTensorBase_nb_binary_operation(a, b, scalar_truediv); }
static PyObject *PyTensorBase_nb_power(PyObject *a, PyObject *b, PyObject *c) = 0;
static PyObject *PyTensorBase_nb_negative(PyObject *a) { return PyTensorBase_nb_unary_operation(a, scalar_negative); }
static PyObject *PyTensorBase_nb_positive(PyObject *a) { return PyTensorBase_nb_unary_operation(a, scalar_positive); }
static PyObject *PyTensorBase_nb_absolute(PyObject *a) { return PyTensorBase_nb_unary_operation(a, scalar_absolute); }

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
/*
TODO: Implement the following instance methods:

* Both in-place and out-of-place
    abs_, cos_, sin_, tan_, log_, exp_, pow_,
    reshape_, fill_, sigmoid_, tanh_, zero_

* Only out-of-place:
    max, min, mean, sum, broadcast_to, permute, transpose/T
*/
static PyMethodDef PyTensorBase_instance_methods[] = {
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

// ----------------------------------------------------------------
// ▗▖ ▗▖       █  ▗▄▖    █         █
// ▐▌ ▐▌ ▐▌    ▀  ▝▜▌    ▀   ▐▌    ▀
// ▐▌ ▐▌▐███  ██   ▐▌   ██  ▐███  ██   ▟█▙ ▗▟██▖
// ▐▌ ▐▌ ▐▌    █   ▐▌    █   ▐▌    █  ▐▙▄▟▌▐▙▄▖▘
// ▐▌ ▐▌ ▐▌    █   ▐▌    █   ▐▌    █  ▐▛▀▀▘ ▀▀█▖
// ▝█▄█▘ ▐▙▄ ▗▄█▄▖ ▐▙▄ ▗▄█▄▖ ▐▙▄ ▗▄█▄▖▝█▄▄▌▐▄▄▟▌
//  ▝▀▘   ▀▀ ▝▀▀▀▘  ▀▀ ▝▀▀▀▘  ▀▀ ▝▀▀▀▘ ▝▀▀  ▀▀▀
// ----------------------------------------------------------------

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

static PyTensorBase *PyTensorBase_create(ShapeArray shape)
{
    PyTensorBase *result = (PyTensorBase *)PyObject_New(PyTensorBase, &PyTensorBaseType);
    if (result == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create new TensorBase object.");
        return NULL;
    }

    if (TensorBase_init(&result->tb, shape) < 0)
    {
        PyErr_SetString(PyExc_RuntimeError, "Failed to initialize tensor base.");
        return NULL;
    }

    return result;
}

// static PyTensorBase *PyTensorBase_shallow_broadcast(PyTensorBase *t, ShapeArray shape)
// {
//     PyTensorBase *result = (PyTensorBase *)PyObject_New(PyTensorBase, &PyTensorBaseType);
//     if (result == NULL)
//     {
//         PyErr_SetString(PyExc_RuntimeError, "Failed to create new PyTensorBase object.");
//         return NULL;
//     }

//     result->tb = t->tb;

//     // Muck with strides...

//     return result;
// }

// ----------------------------------------------------------------
// ▗▄▄▄▖                              ▗▄ ▄▖          ▗▖
// ▝▀█▀▘                              ▐█ █▌      ▐▌  ▐▌
//   █   ▟█▙ ▐▙██▖▗▟██▖ ▟█▙  █▟█▌     ▐███▌ ▟██▖▐███ ▐▙██▖
//   █  ▐▙▄▟▌▐▛ ▐▌▐▙▄▖▘▐▛ ▜▌ █▘       ▐▌█▐▌ ▘▄▟▌ ▐▌  ▐▛ ▐▌
//   █  ▐▛▀▀▘▐▌ ▐▌ ▀▀█▖▐▌ ▐▌ █        ▐▌▀▐▌▗█▀▜▌ ▐▌  ▐▌ ▐▌
//   █  ▝█▄▄▌▐▌ ▐▌▐▄▄▟▌▝█▄█▘ █        ▐▌ ▐▌▐▙▄█▌ ▐▙▄ ▐▌ ▐▌
//   ▀   ▝▀▀ ▝▘ ▝▘ ▀▀▀  ▝▀▘  ▀        ▝▘ ▝▘ ▀▀▝▘  ▀▀ ▝▘ ▝▘
// ----------------------------------------------------------------

static PyObject *PyTensorBase_add_tensor_scalar(PyTensorBase *t, scalar s)
{
    PyTensorBase *result = PyTensorBase_create(t->tb.shape);
    if (!result)
    {
        // NOTE: error string set in PyTensorBase_create
        return NULL;
    }

    TensorBase_add_tensor_scalar(&t->tb, s, &result->tb);
    return (PyObject *)result;
}

static PyObject *PyTensorBase_add_tensor_tensor(PyTensorBase *a, PyTensorBase *b)
{
    PyTensorBase *result = PyTensorBase_create(a_temp.shape);
    if (!result)
    {
        // NOTE: error string set in PyTensorBase_create
        return NULL;
    }
    scalar (*add)(scalar, scalar) = [](scalar a, scalar b)
    { return a + b; };
    if (TensorBase_binary_op_tensorbase_tensorbase(&a->tb, &b->tb, &result->tb, add) == -1)
    {
        // NOTE: error string set in PyTensorBase_create
        return NULL;
    }
    return (PyObject *)result;
}

static PyObject *PyTensorBase_add(PyObject *a, PyObject *b)
{
    // Valid types: PyTensorBase (with broadcastable dimensions), integers, floats
    // TODO: just get types and compare?

    if (!(can_math(a) && can_math(b)))
    {
        PyErr_SetString(PyExc_ValueError, "Invalid types for addition.");
        return NULL;
    }

    // PyTensorBase + (Long | Float)
    if (PyTensorBase_Check(a) && PyFloatOrLong_Check(b))
    {
        return PyTensorBase_add_tensor_scalar((PyTensorBase *)a, PyFloatOrLong_asDouble(b));
    }
    // (Long | Float) + PyTensorBase
    else if (PyFloatOrLong_Check(a) && PyTensorBase_Check(b))
    {
        return PyTensorBase_add_tensor_scalar((PyTensorBase *)b, PyFloatOrLong_asDouble(a));
    }
    // PyTensorBase + PyTensorBase
    else if (PyTensorBase_Check(a) && PyTensorBase_Check(b))
    {
        return PyTensorBase_add_tensor_tensor((PyTensorBase *)a, (PyTensorBase *)b);
    }
    // Else invalid
    else
    {
        PyErr_SetString(PyExc_ValueError, "Invalid types for addition.");
        return NULL;
    }
}

static PyObject *PyTensorBase_div_scalar_tensor(scalar s, PyTensorBase *t)
{
    PyTensorBase *result = PyTensorBase_create(t->tb.shape);
    if (!result)
    {
        // NOTE: error string set in PyTensorBase_create
        return NULL;
    }

    TensorBase_div_scalar_tensor(s, &t->tb, &result->tb);
    return (PyObject *)result;
}

static PyObject *PyTensorBase_divide(PyObject *a, PyObject *b)
{
    if (!(can_math(a) && can_math(b)))
    {
        PyErr_SetString(PyExc_ValueError, "Invalid types for division.");
        return NULL;
    }

    // (Long | Float) + PyTensorBase
    if (PyFloatOrLong_Check(a) && PyTensorBase_Check(b))
    {
        return PyTensorBase_div_scalar_tensor(PyFloatOrLong_asDouble(a), (PyTensorBase *)b);
    }
    // Else invalid
    else
    {
        PyErr_SetString(PyExc_ValueError, "Invalid types for division.");
        return NULL;
    }
}

static PyObject *PyTensorBase_negate(PyObject *a)
{
    PyTensorBase *result = PyTensorBase_create(((PyTensorBase *)a)->tb.shape);
    if (!result)
    {
        // NOTE: error string set in PyTensorBase_create
        return NULL;
    }

    TensorBase_neg(&((PyTensorBase *)a)->tb, &result->tb);
    return (PyObject *)result;
}

static PyObject *PyTensorBase_matrix_multiply(PyTensorBase *a, PyTensorBase *b)
{
    if (PyTensorBase_Check(a) && PyTensorBase_Check(b))
    {
        ShapeArray new_shape = {0};

        if (TensorBase_get_matrix_multiplication_shape(&a->tb, &b->tb, &new_shape) < 0)
        {
            // printf("a->tb.shape: %ld, %ld\n", a->tb.shape[0], a->tb.shape[1]);
            // printf("b->tb.shape: %ld, %ld\n", b->tb.shape[0], b->tb.shape[1]);
            PyErr_SetString(PyExc_ValueError, "Incompatible shapes for matrix multiplication.");
            return NULL;
        }

        PyTensorBase *result = PyTensorBase_create(new_shape);
        if (!result)
        {
            // NOTE: error string set in PyTensorBase_create
            return NULL;
        }

        TensorBase_matrix_multiply(&a->tb, &b->tb, &result->tb);
        return (PyObject *)result;
    }
    // Else invalid
    else
    {
        PyErr_SetString(PyExc_ValueError, "Invalid types for addition.");
        return NULL;
    }
}

static PyObject *PyTensorBase_ones(PyModuleDef *module, PyObject *args)
{
    ShapeArray tb_shape = {0};
    if (args_to_shape(args, &tb_shape) < 0)
    {
        // NOTE: error message set in args_to_shape
        return NULL;
    }

    PyTensorBase *new_tb = PyTensorBase_create(tb_shape);
    // TODO: increment pointer?
    if (new_tb == NULL)
    {
        // NOTE: error message set in PyTensorBase_create
        return NULL;
    }

    for (long i = 0; i < new_tb->tb.numel; i++)
    {
        new_tb->tb.data[i] = 1;
    }

    return new_tb;
}

static PyObject *PyTensorBase_randn(PyModuleDef *module, PyObject *args)
{
    ShapeArray tb_shape = {0};
    if (args_to_shape(args, &tb_shape) < 0)
    {
        // NOTE: error message set in args_to_shape
        return NULL;
    }

    PyTensorBase *new_tb = PyTensorBase_create(tb_shape);
    if (new_tb == NULL)
    {
        // NOTE: error message set in PyTensorBase_create
        return NULL;
    }

    TensorBase_randn(&new_tb->tb, 0, 1);

    return new_tb;
}

static int args_to_shape(PyObject *args, ShapeArray *tb_shape)
{
    // Parse args as tuple of dimensions (or tuple of tuple of dimensions)
    Py_ssize_t tuple_len = PyTuple_Size(args);

    if (tuple_len == 0)
    {
        PyErr_SetString(PyExc_ValueError, "Tensor must have at least one value.");
        return -1;
    }

    if (tuple_len > MAX_RANK)
    {
        PyErr_SetString(PyExc_ValueError, "Tensor rank exceeds maximum allowed.");
        return -1;
    }

    memset(tb_shape, 0, sizeof(ShapeArray));

    for (long i = 0; i < tuple_len; i++)
    {
        PyObject *item = PyTuple_GetItem(args, i);
        if (!PyLong_Check(item))
        {
            PyErr_SetString(PyExc_ValueError, "Tensor dimensions must be integers.");
            return -1;
        }

        (*tb_shape)[i] = PyLong_AsLong(item);
    }

    return 1;
}

static int PyTensorBase_init(PyTensorBase *self, PyObject *args, PyObject *kwds)
{

    ShapeArray tb_shape = {0};
    if (args_to_shape(args, &tb_shape) < 0)
    {
        // NOTE: error message set in args_to_shape
        return -1;
    }

    if (TensorBase_init(&self->tb, tb_shape) < 0)
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

static PyObject *PyTensorBase_str(PyTensorBase *obj)
{
    // TODO: calculate a reasonable buffer size
    char *str_buffer = malloc(100 * sizeof(char));

    TensorBase_to_string(&obj->tb, str_buffer, 100 * sizeof(char));

    return Py_BuildValue("s", str_buffer);
}
