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
    SCALAR_POWER,
    SCALAR_EQ,
    SCALAR_LT,
    SCALAR_GT,
    SCALAR_NEQ,
    SCALAR_LEQ,
    SCALAR_GEQ
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

// Enum for status codes
// TODO: Prefix status codes with `TB_` or `TensorBase_`; Be consistent with naming
typedef enum
{
    OK,
    MALLOC_ERR,
    NULL_INPUT_ERR,
    INCOMPATABLE_BROASCAST_SHAPES,
    INVALID_DIMENSION,
    INVALID_DIMENSION_SIZE,
    DUPLICATE_AGGREGATION_DIM,
    NDIM_OUT_OF_BOUNDS,
    NOT_IMPLEMENTED,
    MATMUL_SINGLETON,
    MATMUL_INCOMPATABLE_SHAPES,
    PERMUTATION_INCORRECT_NDIM,
    PERMUTATION_DUPLICATE_DIM,
    RESHAPE_INVALID_SHAPE_NUMEL_MISMATCH,
    ITEM_NUMEL_NOT_ONE,
    INDEX_OUT_OF_BOUNDS,
    STATUS_SUBSCRIPT_INVALID_PARAMETER,
    INVALID_SHAPES_FOR_SET,
    INVALID_OPERATION,
} StatusCode;

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

// The type of indexing primitive used to access data at a particular dimension of a tensor.
typedef enum
{
    INDEX, // e.g., t[0, 3, 4]
    SLICE  // e.g., t[0:5:2].
} IndexType;

// Definition of an indexing slice object for implementing getitem and setitem.
typedef struct _TensorBase_Subscript
{
    // The type of indexing primitive used to access data at a particular dimension of a tensor.
    IndexType type;
    // If the type of index is INDEX, start is used as the index at the dimension.
    // If the type of the index in SLICE, the fields start, stop, step are interpreted as a Python Slice object.
    // Negative indices are not supported.
    long start;
    long stop;
    long step;
} TensorBaseSubscript;

typedef TensorBaseSubscript SubscriptArray[MAX_RANK];  


// TODO: Refactor code to calculate ndim in methods instead of passing in ndim to function parameters to increase reliability.

/*********************************************************
 *                    Alloc & Dealloc                    *
 *********************************************************/

EXPORT StatusCode TensorBase_init(TensorBase *tb, ShapeArray shape, long ndim);
EXPORT void TensorBase_dealloc(TensorBase *tb);

/*********************************************************
 *                     String Methods                    *
 *********************************************************/

// EXPORT void TensorBase_to_string(TensorBase *td, char *buffer, size_t buffer_size);
EXPORT void TensorBase_to_string(TensorBase *tb);

/*********************************************************
 *                     Braodcasting                      *
 *********************************************************/

EXPORT StatusCode TensorBase_broadcast_to(TensorBase *in, ShapeArray target_shape, long target_ndim, TensorBase *out);
EXPORT StatusCode TensorBase_unbroadcast(TensorBase *in, ShapeArray target_shape, long target_ndim, TensorBase *out);

/*********************************************************
 *                    Linear Algebra                     *
 *********************************************************/

EXPORT StatusCode TensorBase_binary_op_tensorbase_tensorbase(TensorBase *a, TensorBase *b, TensorBase *out, BinaryScalarOperation binop);
EXPORT StatusCode TensorBase_binary_op_tensorbase_scalar(TensorBase *a, scalar s, TensorBase *out, BinaryScalarOperation binop);
EXPORT StatusCode TensorBase_binary_op_scalar_tensorbase(TensorBase *a, scalar s, TensorBase *out, BinaryScalarOperation binop);

EXPORT StatusCode TensorBase_unary_op_inplace(TensorBase *in, UnaryScalarOperation uop);
EXPORT StatusCode TensorBase_unary_op(TensorBase *in, TensorBase *out, UnaryScalarOperation uop);

EXPORT StatusCode TensorBase_get_matrix_multiplication_shape(TensorBase *a, TensorBase *b, ShapeArray *out);
EXPORT StatusCode TensorBase_matrix_multiply(TensorBase *a, TensorBase *b, TensorBase *out);

/*********************************************************
 *                      Aggregation                      *
 *********************************************************/

EXPORT StatusCode TensorBase_aggregate(TensorBase *in, IndexArray aggregation_dimensions, int keepdim, TensorBase *out, AggScalarOperation agg);

/*********************************************************
 *                     Manipulation                      *
 *********************************************************/

EXPORT StatusCode TensorBase_permute(TensorBase *in, IndexArray permutation, long ndim, TensorBase *out);

EXPORT StatusCode TensorBase_reshape_inplace(TensorBase *in, ShapeArray shape, long ndim);
EXPORT StatusCode TensorBase_reshape(TensorBase *in, TensorBase *out, ShapeArray shape, long ndim);

EXPORT StatusCode TensorBase_fill_(TensorBase *in, scalar fill_value);
EXPORT StatusCode TensorBase_randn_(TensorBase *in, scalar mu, scalar sigma);

EXPORT StatusCode TensorBase_item(TensorBase *t, scalar *item);

/*********************************************************
 *                     Set and Get                       *
 *********************************************************/

EXPORT StatusCode TensorBase_get(TensorBase *in, SubscriptArray subscripts, long num_subscripts, TensorBase *out);

EXPORT StatusCode TensorBase_set_scalar(TensorBase *in, SubscriptArray subscripts, long num_subscripts, scalar s);
EXPORT StatusCode TensorBase_set_tensorbase(TensorBase *in, SubscriptArray subscripts, long num_subscripts, TensorBase *t);