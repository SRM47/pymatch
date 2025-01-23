#pragma once
#include <stdlib.h>

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

// Definition of a TensorBase struct.
typedef struct _TensorBase
{
    long numel;          // Number of elements (length of data array)
    long ndim;           // Number of dimensions
    ShapeArray shape;    // Shape of tensor (-1 indicates end of array)
    StrideArray strides; // Strides of tensor (0 indicates end of array)
    scalar *data;        // Tensor data
} TensorBase;

/********************************************************* 
 *                    Alloc & Dealloc                    *
 *********************************************************/

static int TensorBase_init(TensorBase *td, ShapeArray shape, long ndim);
static void TensorBase_dealloc(TensorBase *td);
static int TensorBase_create_empty_like(TensorBase *in, TensorBase *out);

/********************************************************* 
 *                        Utility                        *
 *********************************************************/

static inline int TensorBase_is_singleton(TensorBase *t);
static inline int TensorBase_compare_shape(ShapeArray a_shape, ShapeArray b_shape);

/********************************************************* 
 *                     String Methods                    *
 *********************************************************/

void TensorBase_to_string(TensorBase *td, char *buffer, size_t buffer_size);

/********************************************************* 
 *                     Braodcasting                      *
 *********************************************************/

static int TensorBase_get_broadcast_shape(TensorBase *a, TensorBase *b, ShapeArray broadcast_shape, int *broadcast_ndim);
static int TensorBase_can_broadcast(TensorBase *in, ShapeArray broadcast_shape, int *broadcast_ndim);
static int TensorBase_broadcast_to(TensorBase *in, ShapeArray broadcast_shape, int *broadcast_ndim, TensorBase *out);

/********************************************************* 
 *                    Linear Algebra                     *
 *********************************************************/

static int TensorBase_binary_op_tensorbase_tensorbase(TensorBase *a, TensorBase *b, TensorBase *out, scalar (*op)(scalar, scalar));
static int TensorBase_binary_op_tensorbase_scalar(TensorBase *a, scalar s, TensorBase *out, scalar (*op)(scalar, scalar));
static int TensorBase_binary_op_scalar_tensorbase(TensorBase *a, scalar s, TensorBase *out, scalar (*op)(scalar, scalar));

static int TensorBase_unary_op_inplace(TensorBase *in, scalar (*op)(scalar));
static int TensorBase_unary_op(TensorBase *in, TensorBase *out, scalar (*op)(scalar));

static int TensorBase_get_matrix_multiplication_shape(TensorBase *a, TensorBase *b, ShapeArray *out);
static void TensorBase_matrix_multiply(TensorBase *a, TensorBase *b, TensorBase *out);

/********************************************************* 
 *                      Aggregation                      *
 *********************************************************/

static int TensorBase_aggregate(TensorBase *in, long dim, int keepdim, TensorBase *out, scalar (*aggregate)(scalar*, long));

/********************************************************* 
 *                     Manipulation                      *
 *********************************************************/

static int TensorBase_permute_inplace(TensorBase *in, IndexArray permutation);
static int TensorBase_permute(TensorBase *in, IndexArray permutation, TensorBase *out);

static int TensorBase_reshape_inplace(TensorBase *in, ShapeArray shape);
static int TensorBase_reshape(TensorBase *in, ShapeArray shape, TensorBase *out);

static int TensorBase_fill_(TensorBase *in, scalar fill_value);

 




