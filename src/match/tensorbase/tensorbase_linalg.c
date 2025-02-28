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

StatusCode TensorBase_binary_op_tensorbase_scalar(TensorBase *a, scalar s, TensorBase *out, BinaryScalarOperation binop)
{
    RETURN_IF_ERROR(TensorBase_create_empty_like(a, out));

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
        scalar *a_data = a->data;
        scalar *out_data = out->data;
        for (long i = 0; i < out->numel; i++)
        {
            apply_binop(binop, a_data[i], s, out_data + i); // out->data[i] = a->data[i] `op` s;
        }
    }
    return OK;
}

StatusCode TensorBase_binary_op_scalar_tensorbase(TensorBase *a, scalar s, TensorBase *out, BinaryScalarOperation binop)
{
    RETURN_IF_ERROR(TensorBase_create_empty_like(a, out));

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
        scalar *a_data = a->data;
        scalar *out_data = out->data;
        for (long i = 0; i < out->numel; i++)
        {
            apply_binop(binop, s, a_data[i], out_data + i); // out->data[i] = s `op` a->data[i];
        }
    }
    return OK;
}

StatusCode TensorBase_binary_op_tensorbase_tensorbase(TensorBase *a, TensorBase *b, TensorBase *out, BinaryScalarOperation binop)
{
    // If `a` is a singleton, perform operation as if `a` is a scalar.
    if (TensorBase_is_singleton(a))
    {
        scalar s;
        memcpy(&s, &(a->data), sizeof(scalar));
        return TensorBase_binary_op_scalar_tensorbase(b, s, out, binop);
    }

    // If `b` is a singleton, perform operation as if `b` is a scalar.
    if (TensorBase_is_singleton(b))
    {
        scalar s;
        memcpy(&s, &(b->data), sizeof(scalar));
        return TensorBase_binary_op_tensorbase_scalar(a, s, out, binop);
    }

    if (TensorBase_same_shape(a->shape, b->shape))
    {
        // They have the same shape, so no broadcasting is required.
        RETURN_IF_ERROR(TensorBase_create_empty_like(a, out));

        scalar *a_data = a->data;
        scalar *b_data = b->data;
        scalar *out_data = out->data;

        for (long i = 0; i < out->numel; i++)
        {
            apply_binop(binop, a_data[i], b_data[i], out_data + i); // out->data[i] = a->data[i] `op` b->data[i];
        }
    }
    else
    {
        // Tensors that do not have the same shape must at least be broadcastable.
        ShapeArray broadcasted_tensor_shape;
        long broadcasted_tensor_ndim;
        RETURN_IF_ERROR(TensorBase_get_broadcast_shape(a->shape, a->ndim, b->shape, b->ndim, broadcasted_tensor_shape, &broadcasted_tensor_ndim));

        RETURN_IF_ERROR(TensorBase_init(out, broadcasted_tensor_shape, broadcasted_tensor_ndim));
        // Loop through each element in the broadcasted tensor's data.
        for (long broadcasted_data_index = 0; broadcasted_data_index < out->numel; broadcasted_data_index++)
        {
            // Calculate the corresponding data index in each of the input tensors data array.
            long a_data_index, b_data_index;
            TensorBase_get_translated_data_indices_from_broadcasted_index(
                /* a_shape= */ a->shape,
                /* a_strides= */ a->strides,
                /* a_ndim= */ a->ndim,
                /* b_shape= */ b->shape,
                /* b_strides= */ b->strides,
                /* b_ndim= */ b->ndim,
                /* broadcasted_shape= */ broadcasted_tensor_shape,
                /* broadcasted_ndim= */ broadcasted_tensor_ndim,
                /* broadcasted_data_index= */ broadcasted_data_index,
                &a_data_index,
                &b_data_index);

            apply_binop(binop, a->data[a_data_index], b->data[b_data_index], out->data + broadcasted_data_index);
        }
    }

    return OK;
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

StatusCode TensorBase_unary_op_inplace(TensorBase *in, UnaryScalarOperation uop)
{
    if (in->data == NULL)
    {
        return NULL_INPUT_ERR;
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
    return OK;
}

StatusCode TensorBase_unary_op(TensorBase *in, TensorBase *out, UnaryScalarOperation uop)
{
    if (in == NULL || out == NULL)
    {
        return NULL_INPUT_ERR;
    }

    RETURN_IF_ERROR(TensorBase_create_empty_like(in, out));

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
    return OK;
}

StatusCode TensorBase_initialize_for_matrix_multiplication(TensorBase *a, TensorBase *b, TensorBase *out)
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
    StrideArray strides;
    long ndim;
    long numel;

    if (a->ndim == 1 && b->ndim == 1)
    {
        if (a->numel != b->numel)
        {
            return MATMUL_INCOMPATABLE_SHAPES;
        }
        ndim = 0;
        numel = 1;
    }
    else if (a->ndim == 1 && b->ndim == 2)
    {
        if (a->shape[0] != b->shape[0])
        {
            return MATMUL_INCOMPATABLE_SHAPES;
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
            return MATMUL_INCOMPATABLE_SHAPES;
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
            return MATMUL_INCOMPATABLE_SHAPES;
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
        if (out->data == NULL)
        {
            return MALLOC_ERR;
        }
    }
    return OK;
}

StatusCode TensorBase_matrix_multiply(TensorBase *a, TensorBase *b, TensorBase *out)
{
    if (a == NULL || b == NULL || out == NULL)
    {
        return NULL_INPUT_ERR;
    }

    RETURN_IF_ERROR(TensorBase_initialize_for_matrix_multiplication(a, b, out));

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
        matrix_multiply_2d(a->data, b->data, 1, a->shape[0], b->shape[1], out->data);
    }
    else if (a->ndim == 2 && b->ndim == 1)
    {
        matrix_multiply_2d(a->data, b->data, a->shape[0], b->shape[0], 1, out->data);
    }
    else if (a->ndim == 2 && b->ndim == 2)
    {
        matrix_multiply_2d(a->data, b->data, a->shape[0], a->shape[1], b->shape[1], out->data);
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
            long a_data_index, b_data_index;
            TensorBase_get_translated_data_indices_from_broadcasted_index(
                /* a_shape= */ a->shape,
                /* a_strides= */ a->strides,
                /* a_ndim= */ batch_dims_a,
                /* b_shape= */ b->shape,
                /* b_strides= */ b->strides,
                /* b_ndim= */ batch_dims_b,
                /* broadcasted_shape= */ out->shape,
                /* broadcasted_ndim= */ batch_dims,
                /* broadcasted_data_index= */ broadcasted_data_index,
                &a_data_index,
                &b_data_index);

            matrix_multiply_2d(a->data + a_data_index, b->data + b_data_index, n, l, m, out->data + broadcasted_data_index * n * m);
        } 
    }

    return OK;
}