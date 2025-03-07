#pragma once

#include "tensorbase.h"

// C macro do{}while(0).
#define RETURN_IF_ERROR(x) ({ StatusCode _status = x; if (_status != OK) { return _status; } })

static inline long max_long(long a, long b)
{
    return a > b ? a : b;
}

static inline long min_long(long a, long b)
{
    return a < b ? a : b;
}

static inline int TensorBase_is_singleton(TensorBase *t)
{
    return t->ndim == 0;
}

static inline int TensorBase_same_shape(ShapeArray a_shape, ShapeArray b_shape)
{
    return memcmp(a_shape, b_shape, MAX_RANK * sizeof(long)) == 0;
}

static StatusCode TensorBase_create_empty_like(TensorBase *in, TensorBase *out)
{
    // Assumes out->data doesn't point to any alocated memory.
    if (in == NULL || out == NULL)
    {
        return NULL_INPUT_ERR; // Invalid input or output tensor
    }

    memcpy(out, in, sizeof(TensorBase));

    if (!TensorBase_is_singleton(in))
    {
        out->data = (scalar *)malloc(in->numel * sizeof(scalar));
        if (out->data == NULL)
        {
            return MALLOC_ERR;
        }
    }

    return OK;
}

static StatusCode TensorBase_convert_indices_to_data_index(TensorBase *in, IndexArray coord, long *data_index)
{
    long temporary_index = 0;
    for (long dim = 0; dim < in->ndim; dim++)
    {
        temporary_index += in->strides[dim] * coord[dim];
    }
    *data_index = temporary_index;
    return OK;
}

static void print_long_list(const long *list, size_t size)
{
    printf("[");
    for (long i = 0; i < size; i++)
    {
        printf("%ld, ", list[i]);
    }
    printf("]\n");
}

static StatusCode TensorBase_can_broadcast(ShapeArray source_shape, long source_ndim, ShapeArray target_shape, long target_ndim)
{
    if (source_ndim > target_ndim)
    {
        return INCOMPATABLE_BROASCAST_SHAPES;
    }

    // Broadcast dimension initializations.
    long source_dimension = source_ndim - 1;
    long target_dimension = target_ndim - 1;

    while (source_dimension >= 0)
    {
        if (source_shape[source_dimension] == 1 || target_shape[target_dimension] == 1 || source_shape[source_dimension] == target_shape[target_dimension])
        {
            continue;
        }
        else
        {
            return INCOMPATABLE_BROASCAST_SHAPES;
        }
        source_dimension--;
        target_dimension--;
    }

    return OK;
}

static StatusCode TensorBase_get_broadcast_shape(ShapeArray a_shape, long a_ndim, ShapeArray b_shape, long b_ndim, ShapeArray broadcasted_shape, long *broadcast_ndim)
{
    // Initialize broadcast_shape with -1 to indicate dimensions that haven't been determined yet.
    for (long i = 0; i < MAX_RANK; i++)
    {
        broadcasted_shape[i] = -1;
    }

    // Determine the maximum rank (number of dimensions) between the two provided shapes.
    *broadcast_ndim = max_long(a_ndim, b_ndim);

    // Broadcast dimensions
    long a_dim = a_ndim - 1;
    long b_dim = b_ndim - 1;
    long out_dim = *broadcast_ndim - 1;

    while (out_dim >= 0)
    {
        bool a_dimension_is_one = (a_dim >= 0 && a_shape[a_dim] == 1);
        bool a_dimension_out_of_bounds = (a_dim < 0 && b_dim >= 0);

        bool b_dimension_is_one = (b_dim >= 0 && b_shape[b_dim] == 1);
        bool b_dimension_out_of_bounds = (b_dim < 0 && a_dim >= 0);

        bool dimensions_match = (a_dim >= 0 && b_dim >= 0) && (a_shape[a_dim] == b_shape[b_dim]);

        if (a_dimension_is_one || a_dimension_out_of_bounds)
        {
            broadcasted_shape[out_dim] = b_shape[b_dim];
        }
        else if (b_dimension_is_one || b_dimension_out_of_bounds)
        {
            broadcasted_shape[out_dim] = a_shape[a_dim];
        }
        else if (dimensions_match)
        {
            broadcasted_shape[out_dim] = a_shape[a_dim]; // Equivalently, broadcasted_shape[out_dim] = b_shape[b_dim];
        }
        else
        {
            return INCOMPATABLE_BROASCAST_SHAPES;
        }

        a_dim--;
        b_dim--;
        out_dim--;
    }

    return OK;
}

static void TensorBase_get_translated_data_indices_from_broadcasted_index(
    ShapeArray a_shape,
    StrideArray a_strides,
    long a_ndim,
    ShapeArray b_shape,
    StrideArray b_strides,
    long b_ndim,
    ShapeArray broadcasted_shape,
    long broadcasted_ndim,
    long broadcasted_data_index,
    long *a_data_index,
    long *b_data_index)
{
    // Assumes a_shape, b_shape are broadcastable and the broadcasted shape is broadcasted_shape.
    // Calculate the corresponding data index in each of the input tensors data array.

    long a_dim = a_ndim - 1;
    long b_dim = b_ndim - 1;
    long broadcast_dim = broadcasted_ndim - 1;

    long a_data_index_local_ = 0;
    long b_data_index_local_ = 0;
    while (broadcast_dim >= 0)
    {
        long broadcast_coordinate_at_curr_dim = broadcasted_data_index % broadcasted_shape[broadcast_dim];
        broadcasted_data_index /= broadcasted_shape[broadcast_dim];

        // Maps the broadcasted coordinate back to the input tensor's coordinate.
        // If the input tensor's dimension size is 1, any coordinate in the corresponding broadcasted dimension maps to 0 in the input tensor.
        // Otherwise, if the input tensor's dimension size is greater than 1, the broadcasted coordinate is directly used as the input tensor's coordinate.

        // Check if dimension 'a_dim' contributes to the broadcasted tensor.
        bool a_dimension_is_included = (a_dim >= 0 && a_shape[a_dim] > 1);
        // Translate the broadcasted coordinate to the corresponding coordinate in 'a'.
        long a_translated_coordinate = a_dimension_is_included ? broadcast_coordinate_at_curr_dim : 0;

        // Check if dimension 'b_dim' contributes to the broadcasted tensor.
        bool b_dimension_is_included = (b_dim >= 0 && b_shape[b_dim] > 1);
        // Translate the broadcasted coordinate to the corresponding coordinate in 'b'.
        long b_translated_coordinate = b_dimension_is_included ? broadcast_coordinate_at_curr_dim : 0;

        a_data_index_local_ += a_translated_coordinate * a_strides[a_dim];
        b_data_index_local_ += b_translated_coordinate * b_strides[b_dim];

        broadcast_dim--;
        a_dim--;
        b_dim--;
    }

    *a_data_index = a_data_index_local_;
    *b_data_index = b_data_index_local_;
}

static void matrix_multiply_2d(scalar *A, scalar *B, long n, long l, long m, scalar *out)
{
    // Assumes A is a n x l matrix
    // Assumes B is a l x m matrix
    // out = A@B will be a n x m matrix
    // Assumes out is already allocated
    for (long i = 0; i < n; i++)
    {
        for (long j = 0; j < m; j++)
        {
            scalar sum = 0;
            for (long k = 0; k < l; k++)
            {
                sum += A[i * l + k] * B[k * m + j];
            }
            out[i * m + j] = sum;
        }
    }
}

static inline void apply_binop(BinaryScalarOperation binop, scalar a, scalar b, scalar *result)
{
    switch (binop)
    {
    case SCALAR_ADD:
        *result = a + b;
        break;
    case SCALAR_SUB:
        *result = a - b;
        break;
    case SCALAR_MULT:
        *result = a * b;
        break;
    case SCALAR_FLOORDIV:
        *result = floor(a / b);
        break;
    case SCALAR_TRUEDIV:
        *result = a / b;
        break;
    case SCALAR_POWER:
        *result = pow(a, b);
        break;
    case SCALAR_EQ:
        *result = (a == b);
        break;
    case SCALAR_LT:
        *result = (a < b);
        break;
    case SCALAR_GT:
        *result = (a > b);
        break;
    case SCALAR_NEQ:
        *result = (a != b);
        break;
    case SCALAR_LEQ:
        *result = (a <= b);
        break;
    case SCALAR_GEQ:
        *result = (a >= b);
        break;
    default:
        break;
    }
}

static inline void apply_uop(UnaryScalarOperation uop, scalar a, scalar *result)
{
    switch (uop)
    {
    case SCALAR_NEGATIVE:
        *result = -a;
        break;
    case SCALAR_ABSOLUTE:
        *result = fabs(a);
        break;
    case SCALAR_COS:
        *result = cos(a);
        break;
    case SCALAR_SIN:
        *result = sin(a);
        break;
    case SCALAR_TAN:
        *result = tan(a);
        break;
    case SCALAR_TANH:
        *result = tanh(a);
        break;
    case SCALAR_LOG:
        *result = log(a);
        break;
    case SCALAR_EXP:
        *result = exp(a);
        break;
    case SCALAR_SIGMOID:
        *result = 1.0 / (1.0 + exp(-a));
        break;
    case SCALAR_RELU:
        *result = fmax(0, a);
        break;
    default:
        break;
    }
}

static StatusCode TensorBase_initialize_for_matrix_multiplication(TensorBase *a, TensorBase *b, TensorBase *out)
{
    if (a == NULL || b == NULL || out == NULL)
    {
        return NULL_INPUT_ERR;
    }

    if (TensorBase_is_singleton(a) || TensorBase_is_singleton(b))
    {
        return MATMUL_SINGLETON;
    }

    ShapeArray shape;
    long ndim;

    if (a->ndim == 1 && b->ndim == 1)
    {
        if (a->numel != b->numel)
        {
            return MATMUL_INCOMPATABLE_SHAPES;
        }
        ndim = 0;
    }
    else if (a->ndim == 1 && b->ndim == 2)
    {
        if (a->shape[0] != b->shape[0])
        {
            return MATMUL_INCOMPATABLE_SHAPES;
        }
        shape[0] = b->shape[1];
        ndim = 1;
    }
    else if (a->ndim == 2 && b->ndim == 1)
    {
        if (a->shape[1] != b->shape[0])
        {
            return MATMUL_INCOMPATABLE_SHAPES;
        }
        shape[0] = a->shape[0];
        ndim = 1;
    }
    else if (a->ndim == 2 && b->ndim == 2)
    {
        if (a->shape[1] != b->shape[0])
        {
            return MATMUL_INCOMPATABLE_SHAPES;
        }
        shape[0] = a->shape[0];
        shape[1] = b->shape[1];
        ndim = 2;
    }
    else
    { // are shapes compatible.
        long matrix_dims_a = a->ndim > 1 ? 2 : 1;
        long matrix_dims_b = b->ndim > 1 ? 2 : 1;
        long batch_dims_a = a->ndim - matrix_dims_a; // Number of non matrix dimensions in a
        long batch_dims_b = b->ndim - matrix_dims_b; // Number of non-matrix dimensions in b

        // matrix dimensions begin at batch_dims + 1
        // if a.shape = [2,3,4,5,6]
        // batch dims are [2,3,4] (batch_dim_a = 3)
        // so matrix dims are [5,6], or shape[batch_dim_a] and shape[batch_dim_a+1].

        if (matrix_dims_a == 1)
        {
            if (a->shape[batch_dims_a] != b->shape[batch_dims_b])
            {
                return MATMUL_INCOMPATABLE_SHAPES;
            }
        }
        else
        {
            if (a->shape[batch_dims_a + 1] != b->shape[batch_dims_b])
            {
                return MATMUL_INCOMPATABLE_SHAPES;
            }
        }

        long non_matrix_dims;
        // Broadcast the non-matrx dimensions.
        RETURN_IF_ERROR(TensorBase_get_broadcast_shape(a->shape, batch_dims_a, b->shape, batch_dims_b, shape, &non_matrix_dims));

        ndim = non_matrix_dims;
        if (matrix_dims_a == 1)
        {
            shape[non_matrix_dims] = b->shape[batch_dims_b + 1];
            ndim += 1;
        }
        else if (matrix_dims_b == 1)
        {
            shape[non_matrix_dims] = a->shape[batch_dims_a];
            ndim += 1;
        }
        else
        {
            shape[non_matrix_dims] = a->shape[batch_dims_a];
            shape[non_matrix_dims + 1] = b->shape[batch_dims_b + 1];
            ndim += 2;
        }
    }

    for (long i = ndim; i < MAX_RANK; i++)
    {
        shape[i] = -1;
    }

    return TensorBase_init(out, shape, ndim);
}

static StatusCode calculate_strides_from_shape(ShapeArray shape, long ndim, StrideArray strides)
{
    // `numel_for_strides` is calculated differently, multiplying only dimensions with sizes greater than zero.
    long numel_for_stride = 1;
    // This distinction is crucial for stride calculation. A dimension of size zero effectively has a stride of 1.
    // To simplify stride computations, we use `numel_for_strides` which treats zero-sized dimensions as contributing a multiplicative factor of 1, rather than 0.
    for (int dim = 0; dim < ndim; dim++)
    {
        long dimension_size = shape[dim];
        if (dimension_size < 0)
        {
            return INVALID_DIMENSION_SIZE;
        }
        if (dimension_size > 0)
        {
            numel_for_stride *= dimension_size;
        }
    }

    // Calculate strides.
    // Strides tell you how many elements you need to skip in memory to move to the next element along a specific dimension.
    // For instance, suppose a tensor with a shape [2,3,4], the strides array will be [12, 4, 1].
    // * The stride of the first dimension (2), is the number of elements (in memory) between [0,0,0] and [1,0,0], which is 12.
    // * The stride of the second dimension (3), is the number of elements (in memory) between [0,1,0] and [0,1,0], which is 4.
    // * The stride of the third (last) dimension (4), is the number of elements (in memory) between [0,0,0] and [0,0,1], which is 1.
    // Computationally, the stride at a specific dimension is the product of all latter dimensions.
    memset(strides, 0, MAX_RANK * sizeof(long)); // Initialize strides to 0.
    long stride = numel_for_stride;              // Start with total element count (excluding zero-sized dims).
    for (long dim = 0; dim < ndim; dim++)
    {
        long dimension_size = shape[dim];
        if (dimension_size > 0)
        {
            stride /= dimension_size;
        }
        strides[dim] = stride;
    }

    return OK;
}

typedef struct
{
    scalar a;
    scalar b;
} randn_pair;

// Box-Muller method for generating normally distributed random numbers.
// https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform#C++
static randn_pair randn(scalar mu, scalar sigma)
{
    scalar two_pi = 2.0 * M_PI;

    scalar u1;
    do
    {
        u1 = (scalar)rand() / (scalar)((unsigned)RAND_MAX + 1);
    } while (u1 == 0);

    scalar u2 = (scalar)rand() / (scalar)((unsigned)RAND_MAX + 1);

    scalar mag = sigma * sqrt(-2.0 * log(u1));

    randn_pair result = {0};
    result.a = mag * cos(two_pi * u2) + mu;
    result.b = mag * sin(two_pi * u2) + mu;

    return result;
}
