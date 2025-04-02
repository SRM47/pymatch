#include "tensorbase.h"
#include "tensorbase_util.c"

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
    return TB_OK;
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
    return TB_OK;
}

StatusCode TensorBase_binary_op_tensorbase_tensorbase(TensorBase *lhs, TensorBase *rhs, TensorBase *out, BinaryScalarOperation binop)
{
    // If `lhs` is lhs singleton, perform operation as if `lhs` is a scalar.
    if (TensorBase_is_singleton(lhs))
    {
        scalar lhs_s;
        memcpy(&lhs_s, &(lhs->data), sizeof(scalar));
        // Yes, the flipped rhs/lhs ordering looks weird here.
        return TensorBase_binary_op_scalar_tensorbase(rhs, lhs_s, out, binop);
    }

    // If `rhs` is a singleton, perform operation as if `rhs` is a scalar.
    if (TensorBase_is_singleton(rhs))
    {
        scalar s;
        memcpy(&s, &(rhs->data), sizeof(scalar));
        return TensorBase_binary_op_tensorbase_scalar(lhs, s, out, binop);
    }

    if (TensorBase_same_shape(lhs->shape, rhs->shape))
    {
        // They have the same shape, so no broadcasting is required.
        RETURN_IF_ERROR(TensorBase_create_empty_like(lhs, out));

        scalar *lhs_data = lhs->data;
        scalar *rhs_data = rhs->data;
        scalar *out_data = out->data;

        for (long i = 0; i < out->numel; i++)
        {
            apply_binop(binop, lhs_data[i], rhs_data[i], out_data + i); // out->data[i] = lhs->data[i] `op` rhs->data[i];
        }
    }
    else
    {
        // Tensors that do not have the same shape must at least be broadcastable.
        ShapeArray broadcasted_tensor_shape;
        long broadcasted_tensor_ndim;
        RETURN_IF_ERROR(TensorBase_get_broadcast_shape(lhs->shape, lhs->ndim, rhs->shape, rhs->ndim, broadcasted_tensor_shape, &broadcasted_tensor_ndim));

        RETURN_IF_ERROR(TensorBase_init(out, broadcasted_tensor_shape, broadcasted_tensor_ndim));
        // Loop through each element in the broadcasted tensor's data.
        for (long broadcasted_data_index = 0; broadcasted_data_index < out->numel; broadcasted_data_index++)
        {
            // Calculate the corresponding data index in each of the input tensors data array.
            long lhs_data_index, rhs_data_index;
            TensorBase_get_translated_data_indices_from_broadcasted_index(
                /* a_shape= */ lhs->shape,
                /* a_strides= */ lhs->strides,
                /* a_ndim= */ lhs->ndim,
                /* b_shape= */ rhs->shape,
                /* b_strides= */ rhs->strides,
                /* b_ndim= */ rhs->ndim,
                /* broadcasted_shape= */ broadcasted_tensor_shape,
                /* broadcasted_ndim= */ broadcasted_tensor_ndim,
                /* broadcasted_data_index= */ broadcasted_data_index,
                &lhs_data_index,
                &rhs_data_index);

            apply_binop(binop, lhs->data[lhs_data_index], rhs->data[rhs_data_index], out->data + broadcasted_data_index);
        }
    }

    return TB_OK;
}

StatusCode TensorBase_unary_op_inplace(TensorBase *in, UnaryScalarOperation uop)
{
    if (in->data == NULL)
    {
        return TB_NULL_INPUT_ERROR;
    }

    if (TensorBase_is_singleton(in))
    {
        // Copy the bits held in in->data into the scalar variable.
        scalar in_value;
        memcpy(&in_value, &(in->data), sizeof(scalar));
        // Apply the unary operation: result = uop(in_value);
        scalar result;
        apply_uop(uop, in_value, &result);
        // Copy the bits of result into out->data.
        memcpy(&(in->data), &result, sizeof(scalar));
    }
    else
    {
        for (long i = 0; i < in->numel; i++)
        {
            apply_uop(uop, in->data[i], in->data + i); // in->data[i] = uop(in->data[i]).
        }
    }
    return TB_OK;
}

StatusCode TensorBase_unary_op(TensorBase *in, TensorBase *out, UnaryScalarOperation uop)
{
    if (in == NULL || out == NULL)
    {
        return TB_NULL_INPUT_ERROR;
    }

    RETURN_IF_ERROR(TensorBase_create_empty_like(in, out));

    if (TensorBase_is_singleton(in))
    {
        // Copy the bits held in in->data into the scalar variable.
        scalar in_value;
        memcpy(&in_value, &(in->data), sizeof(scalar));
        // Apply the unary operation: result = uop(in_value);
        scalar result;
        apply_uop(uop, in_value, &result);
        // Copy the bits of result into bits of the out->data pointer.
        memcpy(&(out->data), &result, sizeof(scalar));
    }
    else
    {
        for (long i = 0; i < out->numel; i++)
        {
            apply_uop(uop, in->data[i], out->data + i); // out->data[i] = uop(in->data[i]).
        }
    }
    return TB_OK;
}

StatusCode TensorBase_matrix_multiply(TensorBase *lhs, TensorBase *rhs, TensorBase *out)
{
    if (lhs == NULL || rhs == NULL || out == NULL)
    {
        return TB_NULL_INPUT_ERROR;
    }

    RETURN_IF_ERROR(TensorBase_initialize_for_matrix_multiplication(lhs, rhs, out));

    if (lhs->ndim == 1 && rhs->ndim == 1)
    {
        // If both tensors are one dimensional, compute the dot product
        scalar sum = 0;
        for (long i = 0; i < lhs->numel; i++)
        {
            sum += lhs->data[i] * rhs->data[i];
        }
        memcpy(&out->data, &sum, sizeof(scalar));
    }
    else if (lhs->ndim == 1 && rhs->ndim == 2)
    {
        // (a) @ (a, b) is interpreted as (1, a) @ (a, b).
        matrix_multiply_2d(lhs->data, rhs->data, 1, lhs->shape[0] /* rhs->shape[0] */, rhs->shape[1], out->data);
    }
    else if (lhs->ndim == 2 && rhs->ndim == 1)
    {
        // (a, b) @ (b) is interpreted as (a, b) @ (b, 1).
        matrix_multiply_2d(lhs->data, rhs->data, lhs->shape[0], rhs->shape[0] /* lhs->shape[1] */, 1, out->data);
    }
    else if (lhs->ndim == 2 && rhs->ndim == 2)
    {
        // (a, b) @ (b, c) is interpreted normally.
        matrix_multiply_2d(lhs->data, rhs->data, lhs->shape[0], lhs->shape[1] /* rhs->shape[0] */, rhs->shape[1], out->data);
    }
    else
    {
        // Get batch and matrix dimensions.
        long matrix_dims_a = lhs->ndim > 1 ? 2 : 1;
        long matrix_dims_b = rhs->ndim > 1 ? 2 : 1;
        long batch_dims_a = lhs->ndim - matrix_dims_a; // Number of non matrix dimensions in lhs
        long batch_dims_b = rhs->ndim - matrix_dims_b; // Number of non-matrix dimensions in rhs

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
            l = lhs->shape[0]; // Same as rhs->shape[batch_dims_b]
            m = rhs->shape[batch_dims_b + 1];
        }
        else if (matrix_dims_b == 1)
        {
            n = lhs->shape[batch_dims_a];
            l = rhs->shape[0]; // Same as lhs->shape[batch_dims_a+1]
            m = 1;
        }
        else
        {
            n = lhs->shape[batch_dims_a];
            l = lhs->shape[batch_dims_a + 1]; // Same as rhs->shape[batch_dims_b]
            m = rhs->shape[batch_dims_b + 1];
        }

        // Loop through each element in the non matrix dimensions broadcasted tensor's data.
        for (long broadcasted_data_index = 0; broadcasted_data_index < numel_in_batch_dims; broadcasted_data_index++)
        {
            // For each element in the data index, calculate the corresponding
            // element in each of the input tensors.
            long lhs_data_index, rhs_data_index;
            TensorBase_get_translated_data_indices_from_broadcasted_index(
                /* a_shape= */ lhs->shape,
                /* a_strides= */ lhs->strides,
                /* a_ndim= */ batch_dims_a,
                /* b_shape= */ rhs->shape,
                /* b_strides= */ rhs->strides,
                /* b_ndim= */ batch_dims_b,
                /* broadcasted_shape= */ out->shape,
                /* broadcasted_ndim= */ batch_dims,
                /* broadcasted_data_index= */ broadcasted_data_index,
                &lhs_data_index,
                &rhs_data_index);

            matrix_multiply_2d(lhs->data + lhs_data_index, rhs->data + rhs_data_index, n, l, m, out->data + broadcasted_data_index * n * m);
        }
    }

    return TB_OK;
}