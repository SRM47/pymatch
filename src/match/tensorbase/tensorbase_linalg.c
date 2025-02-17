#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include "tensorbase.h"
#include "tensorbase_util.c"

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

int TensorBase_binary_op_tensorbase_scalar(TensorBase *a, scalar s, TensorBase *out, BinaryScalarOperation binop)
{
    if (TensorBase_create_empty_like(a, out) == -1)
    {
        return -1;
    }
    if (TensorBase_is_singleton(a))
    {
        // Copy the bits of a->data into a_value.
        scalar a_value;
        memcpy(&a_value, &(a->data), sizeof(scalar));
        // Compute the result of the operator.
        scalar result;
        apply_binop(binop, a_value, s, &result);
        // Copy the bits of result into out->data.
        memcpy(&(out->data), &result, sizeof(scalar));
    }
    else
    {
        for (long i = 0; i < out->numel; i++)
        {
            apply_binop(binop, a->data[i], s, out->data + i);
        }
    }
    return 0;
}

int TensorBase_binary_op_scalar_tensorbase(TensorBase *a, scalar s, TensorBase *out, BinaryScalarOperation binop)
{
    if (TensorBase_create_empty_like(a, out) == -1)
    {
        return -1;
    }
    if (TensorBase_is_singleton(a))
    {
        // Copy the bits of a->data into a_value.
        scalar a_value;
        memcpy(&a_value, &(a->data), sizeof(scalar));
        // Compute the result of the operator.
        scalar result;
        apply_binop(binop, s, a_value, &result);
        // Copy the bits of result into out->data.
        memcpy(&(out->data), &result, sizeof(scalar));
    }
    else
    {
        for (long i = 0; i < out->numel; i++)
        {
            apply_binop(binop, s, a->data[i], out->data + i);
        }
    }
    return 0;
}

int TensorBase_binary_op_tensorbase_tensorbase(TensorBase *a, TensorBase *b, TensorBase *out, BinaryScalarOperation binop)
{
    // Check if a is a singleton.
    if (TensorBase_is_singleton(a))
    {
        scalar s;
        memcpy(&s, &(a->data), sizeof(scalar));
        return TensorBase_binary_op_scalar_tensorbase(b, s, out, binop);
    }

    // Check if b is a singleton.
    if (TensorBase_is_singleton(b))
    {
        scalar s;
        memcpy(&s, &(b->data), sizeof(scalar));
        return TensorBase_binary_op_tensorbase_scalar(a, s, out, binop);
    }

    // Check if the two tensorbase structures don't have the same dimensions
    if (TensorBase_same_shape(a->shape, b->shape))
    {
        // They have the same shape.
        if (TensorBase_create_empty_like(a, out) == -1)
        {
            return -1;
        }
        for (long i = 0; i < out->numel; i++)
        {
            apply_binop(binop, a->data[i], b->data[i], out->data + i);
        }
    }
    else
    {
        // They don't have the same shape must *attempt to* broadcast.
        ShapeArray broadcasted_tensor_shape;
        long broadcasted_tensor_ndim;
        if (TensorBase_get_broadcast_shape(a->shape, a->ndim, b->shape, b->ndim, broadcasted_tensor_shape, &broadcasted_tensor_ndim) == -1)
        {
            // Incompatible broadcasting shapes.
            return -1;
        }

        if (TensorBase_init(out, broadcasted_tensor_shape, broadcasted_tensor_ndim) == -1)
        {
            return -1;
        }
        // Loop through each element in the broadcasted tensor's data.
        for (long broadcasted_data_index = 0; broadcasted_data_index < out->numel; broadcasted_data_index++)
        {
            // For each element in the data index, calculate the corresponding
            // element in each of the input tensors.
            long a_data_index = 0;
            long b_data_index = 0;

            long a_dim = a->ndim - 1;
            long b_dim = b->ndim - 1;
            long broadcast_dim = out->ndim - 1;

            long temp = broadcasted_data_index;

            for (; broadcast_dim >= 0; broadcast_dim--, a_dim--, b_dim--)
            {
                // (2,5,6)
                long broadcast_coordinate_at_curr_dim = temp % out->shape[broadcast_dim];
                temp /= out->shape[broadcast_dim];

                if (a_dim >= 0 && a->shape[a_dim] > 1)
                {
                    a_data_index += broadcast_coordinate_at_curr_dim * a->strides[a_dim];
                }

                if (b_dim >= 0 && b->shape[b_dim] > 1)
                {
                    b_data_index += broadcast_coordinate_at_curr_dim * b->strides[b_dim];
                }
            }

            apply_binop(binop, a->data[a_data_index], b->data[b_data_index], out->data + broadcasted_data_index);
        }
    }

    return 0;
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

int TensorBase_unary_op_inplace(TensorBase *in, UnaryScalarOperation uop)
{
    if (in->data == NULL)
    {
        return -1;
    }
    if (TensorBase_is_singleton(in))
    {
        scalar in_value;
        memcpy(&in_value, &(in->data), sizeof(scalar));

        scalar result;
        apply_uop(uop, in_value, &result);

        memcpy(&(in->data), &result, sizeof(scalar));
    }
    else
    {
        for (long i = 0; i < in->numel; i++)
        {
            apply_uop(uop, in->data[i], in->data + i);
        }
    }
    return 0;
}

int TensorBase_unary_op(TensorBase *in, TensorBase *out, UnaryScalarOperation uop)
{
    if (TensorBase_create_empty_like(in, out) == -1)
    {
        return -1;
    }
    if (TensorBase_is_singleton(in))
    {
        scalar in_value;
        memcpy(&in_value, &(in->data), sizeof(scalar));

        scalar result;
        apply_uop(uop, in_value, &result);

        memcpy(&(out->data), &result, sizeof(scalar));
    }
    else
    {
        for (long i = 0; i < out->numel; i++)
        {
            apply_uop(uop, in->data[i], out->data + i);
        }
    }
    return 0;
}

int TensorBase_initialize_for_matrix_multiplication(TensorBase *a, TensorBase *b, TensorBase *out)
{
    if (a == NULL || b == NULL || out == NULL)
    {
        return -1;
    }

    if (TensorBase_is_singleton(a) || TensorBase_is_singleton(b))
    {
        return -1;
    }

    ShapeArray shape;
    StrideArray strides;
    long ndim;
    long numel;

    if (a->ndim == 1 && b->ndim == 1)
    {
        if (a->numel != b->numel)
        {
            return -1;
        }
        ndim = 0;
        numel = 1;
    }
    else if (a->ndim == 1 && b->ndim == 2)
    {
        if (a->shape[0] != b->shape[0])
        {
            return -1;
        }
        shape[0] = b->shape[1];
        strides[0] = 1;
        ndim = 1;
        numel = b->shape[1];
    }
    else if (a->ndim == 2 && b->ndim == 1)
    {
        if (a->shape[1] != b->shape[0])
        {
            return -1;
        }
        shape[0] = a->shape[0];
        strides[0] = 1;
        ndim = 1;
        numel = a->shape[0];
    }
    else if (a->ndim == 2 && b->ndim == 2)
    {
        if (a->shape[1] != b->shape[0])
        {
            return -1;
        }
        shape[0] = a->shape[0];
        shape[1] = b->shape[1];
        strides[0] = b->shape[1];
        strides[1] = 1;
        ndim = 2;
        numel = a->shape[0] * b->shape[1];
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
                printf("%ld, %ld", batch_dims_a, batch_dims_b);
                return -2;
            }
        }
        else
        {
            if (a->shape[batch_dims_a + 1] != b->shape[batch_dims_b])
            {
                printf("%ld, %ld", batch_dims_a, batch_dims_b);
                return -2;
            }
        }

        long non_matrix_dims;
        // Broadcast the non-matrx dimensions.
        if (TensorBase_get_broadcast_shape(a->shape, batch_dims_a, b->shape, batch_dims_b, shape, &non_matrix_dims) == -1)
        {
            return -2;
        }

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

        // Calculate numel and strides
        // Calculate Tensorbase shape and number of elements.
        long numel_for_stride = 1;
        numel = 1;
        for (int i = 0; i < ndim; i++)
        {
            long dim = shape[i];
            numel *= dim;
            if (dim > 0)
            {
                numel_for_stride *= dim;
            }
        }

        // Calculate Tensorbase strides.
        long stride = numel_for_stride;
        for (long i = 0; i < ndim; i++)
        {
            long dim = shape[i];
            if (dim > 0)
            {
                stride /= dim;
            }
            strides[i] = stride;
        }
    }

    for (long i = ndim; i < MAX_RANK; i++)
    {
        shape[i] = -1;
        strides[i] = 0;
    }
    memcpy(&out->shape, &shape, sizeof(scalar) * MAX_RANK);
    memcpy(&out->strides, &strides, sizeof(scalar) * MAX_RANK);
    out->numel = numel;
    out->ndim = ndim;
    if (ndim > 0)
    {
        out->data = (scalar *)malloc(numel * sizeof(scalar));
    }
    return 0;
}

int matrix_multiply_2d(scalar *A, scalar *B, long n, long l, long m, scalar *out)
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

    return 0;
}

int TensorBase_matrix_multiply(TensorBase *a, TensorBase *b, TensorBase *out)
{
    if (a == NULL || b == NULL || out == NULL)
    {
        return -1;
    }

    int status = TensorBase_initialize_for_matrix_multiplication(a, b, out);
    if (status < 0)
    {
        return status;
    }

    if (a->ndim == 1 && b->ndim == 1)
    {
        scalar sum = 0;
        for (long i = 0; i < a->numel; i++)
        {
            sum += a->data[i] * b->data[i];
        }
        memcpy(&out->data, &sum, sizeof(scalar));
    }
    else if (a->ndim == 1 && b->ndim == 2)
    {
        return matrix_multiply_2d(a->data, b->data, 1, a->shape[0], b->shape[1], out->data);
    }
    else if (a->ndim == 2 && b->ndim == 1)
    {
        return matrix_multiply_2d(a->data, b->data, a->shape[0], b->shape[0], 1, out->data);
    }
    else if (a->ndim == 2 && b->ndim == 2)
    {
        return matrix_multiply_2d(a->data, b->data, a->shape[0], a->shape[1], b->shape[1], out->data);
    }
    else
    {
        // Get batch and matrix dimensions.
        long matrix_dims_a = a->ndim > 1 ? 2 : 1;
        long matrix_dims_b = b->ndim > 1 ? 2 : 1;
        long batch_dims_a = a->ndim - matrix_dims_a; // Number of non matrix dimensions in a
        long batch_dims_b = b->ndim - matrix_dims_b; // Number of non-matrix dimensions in b

        long batch_dims = max_long(batch_dims_a, batch_dims_b);
        long numel_in_batch_dims = 1;
        for (int dim = 0; dim < batch_dims; dim++)
        {
            numel_in_batch_dims *= out->shape[dim];
        }

        // Calculate the shape of the matrices to multiple from the matrix dimensions.
        long n, l, m;

        if (matrix_dims_a == 1)
        {
            n = 1;
            l = a->shape[0]; // Same as b->shape[batch_dims_b]
            m = b->shape[batch_dims_b + 1];
        }
        else if (matrix_dims_b == 1)
        {
            n = a->shape[batch_dims_a];
            l = b->shape[0]; // Same as a->shape[batch_dims_a+1]
            m = 1;
        }
        else
        {
            n = a->shape[batch_dims_a];
            l = a->shape[batch_dims_a + 1]; // Same as b->shape[batch_dims_b]
            m = b->shape[batch_dims_b + 1];
        }

        // Loop through each element in the non matrix dimensions broadcasted tensor's data.
        for (long broadcasted_data_index = 0; broadcasted_data_index < numel_in_batch_dims; broadcasted_data_index++)
        {
            // For each element in the data index, calculate the corresponding
            // element in each of the input tensors.
            long a_data_index = 0;
            long b_data_index = 0;

            long a_dim = batch_dims_a - 1;
            long b_dim = batch_dims_b - 1;
            long broadcast_dim = batch_dims - 1;

            long temp = broadcasted_data_index;

            for (; broadcast_dim >= 0; broadcast_dim--, a_dim--, b_dim--)
            {
                long broadcast_coordinate_at_curr_dim = temp % out->shape[broadcast_dim];
                temp /= out->shape[broadcast_dim];

                if (a_dim >= 0 && a->shape[a_dim] > 1)
                {
                    a_data_index += broadcast_coordinate_at_curr_dim * a->strides[a_dim];
                }

                if (b_dim >= 0 && b->shape[b_dim] > 1)
                {
                    b_data_index += broadcast_coordinate_at_curr_dim * b->strides[b_dim];
                }
            }

            status = matrix_multiply_2d(a->data + a_data_index, b->data + b_data_index, n, l, m, out->data + broadcasted_data_index * n * m);
            if (status < 0)
            {
                return status;
            }
        }
        return 0;
    }
}