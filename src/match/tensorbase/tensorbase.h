#pragma once

#ifdef __GNUC__
#define EXPORT __attribute__((visibility("default")))
#else
#define EXPORT
#endif

/*********************************************************
 *                        GLOBALS                        *
 *********************************************************/

// Tensors will support only float64 elements.
typedef double scalar;

// Tensors have a maximum rank of 8.
#define MAX_RANK 8

// Define type aliases.
typedef long IndexArray[MAX_RANK];
typedef long ShapeArray[MAX_RANK];
typedef long StrideArray[MAX_RANK];

// Enum for binary operations.
typedef enum
{
    SCALAR_ADD,
    SCALAR_SUB,
    SCALAR_MULT,
    SCALAR_FLOORDIV,
    SCALAR_TRUEDIV,
    SCALAR_POWER
} BinaryScalarOperation;

// Enum for unary operations.
typedef enum
{
    SCALAR_NEGATIVE,
    SCALAR_ABSOLUTE,
    SCALAR_COS,
    SCALAR_SIN,
    SCALAR_TAN,
    SCALAR_TANH,
    SCALAR_LOG,
    SCALAR_EXP,
    SCALAR_SIGMOID,
    SCALAR_RELU
} UnaryScalarOperation;

// Enum for aggregation operations (max, min, sum, mean).
typedef enum
{
    SCALAR_AGG_SUM,
    SCALAR_AGG_MEAN
} AggScalarOperation;

// Definition of a TensorBase struct.
typedef struct _TensorBase
{
    long data_ref_count; // Number of references to the data pointer (for deallocation).
    long numel;          // Number of elements (length of data array)
    long ndim;           // Number of dimensions
    ShapeArray shape;    // Shape of tensor (-1 indicates end of array)
    StrideArray strides; // Strides of tensor (0 indicates end of array)
    scalar *data;        // Tensor data
} TensorBase;

/*********************************************************
 *                    Alloc & Dealloc                    *
 *********************************************************/

EXPORT int TensorBase_init(TensorBase *tb, ShapeArray shape, long ndim);
EXPORT void TensorBase_dealloc(TensorBase *tb);

/*********************************************************
 *                     String Methods                    *
 *********************************************************/

// EXPORT void TensorBase_to_string(TensorBase *td, char *buffer, size_t buffer_size);
EXPORT void TensorBase_to_string(TensorBase *td);

/*********************************************************
 *                     Braodcasting                      *
 *********************************************************/

EXPORT int TensorBase_broadcast_to(TensorBase *in, ShapeArray broadcast_shape, int *broadcast_ndim, TensorBase *out);

/*********************************************************
 *                    Linear Algebra                     *
 *********************************************************/

EXPORT int TensorBase_binary_op_tensorbase_tensorbase(TensorBase *a, TensorBase *b, TensorBase *out, BinaryScalarOperation binop);
EXPORT int TensorBase_binary_op_tensorbase_scalar(TensorBase *a, scalar s, TensorBase *out, BinaryScalarOperation binop);
EXPORT int TensorBase_binary_op_scalar_tensorbase(TensorBase *a, scalar s, TensorBase *out, BinaryScalarOperation binop);

EXPORT int TensorBase_unary_op_inplace(TensorBase *in, UnaryScalarOperation uop);
EXPORT int TensorBase_unary_op(TensorBase *in, TensorBase *out, UnaryScalarOperation uop);

EXPORT int TensorBase_get_matrix_multiplication_shape(TensorBase *a, TensorBase *b, ShapeArray *out);
EXPORT int TensorBase_matrix_multiply(TensorBase *a, TensorBase *b, TensorBase *out);

/*********************************************************
 *                      Aggregation                      *
 *********************************************************/

EXPORT int TensorBase_aggregate(TensorBase *in, IndexArray dim, int keepdim, TensorBase *out, AggScalarOperation agg);

/*********************************************************
 *                     Manipulation                      *
 *********************************************************/

EXPORT int TensorBase_permute(TensorBase *in, IndexArray permutation, TensorBase *out);

EXPORT int TensorBase_reshape_inplace(TensorBase *in, ShapeArray shape, long ndim);
EXPORT int TensorBase_reshape(TensorBase *in, TensorBase *out, ShapeArray shape, long ndim);

EXPORT int TensorBase_fill_(TensorBase *in, scalar fill_value);
EXPORT int TensorBase_randn_(TensorBase *in, scalar mu, scalar sigma);

EXPORT int TensorBase_item(TensorBase *t, scalar *item);

EXPORT void print_double_list(const double *list, size_t size);
EXPORT void print_long_list(const long *list, size_t size);
