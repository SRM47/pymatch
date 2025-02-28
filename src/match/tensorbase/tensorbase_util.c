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

static StatusCode TensorBase_convert_indices_to_data_index(TensorBase *in, IndexArray curr_index, long *index)
{
    long res = 0;
    for (long i = 0; i < in->ndim; i++)
    {
        res += in->strides[i] * curr_index[i];
    }
    *index = res;
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
