#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include "tensorbase.h"
#include "tensorbase_util.c"

int TensorBase_aggregate(TensorBase *in, IndexArray dim, int keepdim, TensorBase *out, AggScalarOperation agg)
{
    // Input validation
    if (in == NULL || out == NULL)
    {
        return -1;
    }
    long agg_ndim;
    IndexArray agg_dims;
    memset(agg_dims, 0, sizeof(scalar) * MAX_RANK);
    for (agg_ndim = 0; agg_ndim < MAX_RANK; agg_ndim++)
    {
        if (dim[agg_ndim] < 0)
        {
            break;
        }
        if (dim[agg_ndim] >= in->ndim)
        {
            return -1;
        }
        if (agg_dims[dim[agg_ndim]] != 0)
        {
            return -1; // Duplicate
        }
        agg_dims[dim[agg_ndim]] = 1;
    }

    // If singleton, just return the singleton
    if (TensorBase_is_singleton(in))
    {
        memcpy(out, in, sizeof(TensorBase));
        return 0;
    }

    // If no dims were provided, then aggregate over the entire array
    ShapeArray new_shape;
    long d;
    for (d = 0; d < in->ndim; d++)
    {
        new_shape[d] = (agg_dims[d] == 0) ? in->shape[d] : 1;
    }
    for (; d < MAX_RANK; d++)
    {
        new_shape[d] = -1;
    }

    // Initialize the out tensor with the new shape
    RETURN_IF_ERROR(TensorBase_init(out, new_shape, in->ndim));

    scalar num_agg_elem = in->numel / out->numel;

    // Loop through each element in the input tensors data, and map what position in the aggregated tensors data.
    scalar temp_out_data[out->numel];
    memset(temp_out_data, 0, sizeof(scalar) * out->numel);
    for (long in_data_index = 0; in_data_index < in->numel; in_data_index++)
    {
        // For each element in the data index, calculate the corresponding
        // element in each of the input tensors.
        long out_data_index = 0;

        for (long d = out->ndim - 1, temp = in_data_index; d >= 0; d--)
        {
            // Get the coordinate in the original tensor (in)
            long in_coordinate_at_curr_dim = temp % in->shape[d];
            temp /= in->shape[d];

            // If the current dimension is a summed dimension, then convert it to 0 in the out tensor
            // Add to the index into out data array.
            if (agg_dims[d] == 0)
            {
                // If the current dimension is a sumemd dimension, the converted index would be 0 and 0*strides[d] = 0. So we only multiply is if not sumemd dimension.
                out_data_index += in_coordinate_at_curr_dim * out->strides[d];
            }
        }

        switch (agg)
        {
        case SCALAR_AGG_MEAN:
        case SCALAR_AGG_SUM:
            temp_out_data[out_data_index] += in->data[in_data_index];
            break;
        }
    }

    if (agg == SCALAR_AGG_MEAN)
    {
        for (long i = 0; i < out->numel; i++)
        {
            temp_out_data[i] /= num_agg_elem;
        }
    }

    memcpy(out->data, temp_out_data, out->numel * sizeof(scalar));

    if (keepdim)
    {
        return 0;
    }

    ShapeArray new_shape_after_keepdim;
    long ndim_after_keepdim;

    for (d = 0, ndim_after_keepdim = 0; d < in->ndim; d++)
    {
        if (agg_dims[d] == 0)
        {
            new_shape_after_keepdim[ndim_after_keepdim] = new_shape[d];
            ndim_after_keepdim++;
        }
    }
    for (d = ndim_after_keepdim; d < MAX_RANK; d++)
    {
        new_shape_after_keepdim[d] = -1;
    }

    RETURN_IF_ERROR(TensorBase_reshape_inplace(out, new_shape_after_keepdim, ndim_after_keepdim));

    return 0;
}