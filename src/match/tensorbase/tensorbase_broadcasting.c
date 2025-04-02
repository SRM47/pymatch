#include "tensorbase.h"
#include "tensorbase_util.c"

StatusCode TensorBase_broadcast_to(TensorBase *in, ShapeArray target_shape, long target_ndim, TensorBase *out)
{
    return TB_NOT_IMPLEMENTED_ERROR;
}

StatusCode TensorBase_unbroadcast(TensorBase *in, ShapeArray target_shape, long target_ndim, TensorBase *out)
{
    RETURN_IF_ERROR(TensorBase_create_empty_like(in, out));

    if (!TensorBase_is_singleton(in))
    {
        memcpy(out->data, in->data, sizeof(scalar) * out->numel);
    }

    if (!TensorBase_same_shape(in->shape, target_shape))
    {
        long ndim_diff = labs(in->ndim - target_ndim);

        // Aggregate (sum) the excess dimensions in the input tensor.
        if (ndim_diff != 0)
        {
            IndexArray summation_dimensions;
            for (long dim = 0; dim < ndim_diff; dim++)
            {
                summation_dimensions[dim] = dim;
            }
            for (long dim = ndim_diff; dim < MAX_RANK; dim++)
            {
                summation_dimensions[dim] = -1;
            }

            RETURN_IF_ERROR(TensorBase_aggregate(in, summation_dimensions, 0, out, SCALAR_AGG_SUM));
        }

        // Collate the dimensions in the target unbroadcasted shape that had a size of 1. Then, aggregate (sum) over those dimensions.
        IndexArray originally_ones;
        long ones_dimension_count = 0;
        // Identify dimensions that were originally of size 1.
        for (long i = 0; i < target_ndim; i++)
        {
            if (target_shape[i] == 1)
            {
                originally_ones[ones_dimension_count] = i;
                ones_dimension_count++;
            }
        }
        // If there were dimensions of size 1, perform aggregation.
        if (ones_dimension_count > 0)
        {
            // Pad the remaining indices of originally_ones with -1.
            // This is needed for the TensorBase_aggregate function.
            for (long dim = ones_dimension_count; dim < MAX_RANK; dim++)
            {
                originally_ones[dim] = -1;
            }

            // Temporary tensor to store the result of aggregation.
            TensorBase temp;

            // Aggregate (sum) over the identified dimensions.
            RETURN_IF_ERROR(TensorBase_aggregate(out, originally_ones, 1, &temp, SCALAR_AGG_SUM));

            // At this point, 'out' points to stale data, and 'temp' contains the desired aggregated data.
            // Deallocate the old, stale data in 'out' to prevent memory leaks.
            TensorBase_dealloc(out);
            // Copy the aggregated tensor data from 'temp' to 'out'. Now, 'out' holds the correct aggregated data.
            memcpy(out, &temp, sizeof(TensorBase));
        }
    }
    return TB_OK;
}