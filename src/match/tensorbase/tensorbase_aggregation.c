#include "tensorbase.h"
#include "tensorbase_util.c"

StatusCode TensorBase_aggregate(TensorBase *in, IndexArray aggregation_dimensions, int keepdim, TensorBase *out, AggScalarOperation agg)
{
    if (in == NULL || out == NULL)
    {
        return NULL_INPUT_ERR;
    }

    // This array will act as a "flag" array, where each index corresponds to a dimension of the input tensor.
    // A value of 1 at a specific index means that dimension should be aggregated (reduced),
    // and a 0 means it should be kept.
    IndexArray dimensions_to_aggregate;
    memset(dimensions_to_aggregate, 0, sizeof(scalar) * MAX_RANK);
    for (long i = 0; i < MAX_RANK; i++)
    {
        long dim = aggregation_dimensions[i];
        if (dim < 0)
        {
            break;
        }
        if (dim >= in->ndim)
        {
            return INVALID_DIMENSION;
        }
        if (dimensions_to_aggregate[dim] != 0)
        {
            return DUPLICATE_AGGREGATION_DIM;
        }
        dimensions_to_aggregate[dim] = 1;
    }

    if (TensorBase_is_singleton(in))
    {
        memcpy(out, in, sizeof(TensorBase));
        return OK;
    }

    ShapeArray aggregated_shape;
    for (long dim = 0; dim < in->ndim; dim++)
    {
        if (dimensions_to_aggregate[dim] == 0)
        {
            aggregated_shape[dim] = in->shape[dim];
        }
        else
        {
            aggregated_shape[dim] = 1;
        }
    }
    for (long dim = in->ndim; dim < MAX_RANK; dim++)
    {
        aggregated_shape[dim] = -1;
    }

    // Initialize the output tensor with the new shape.
    RETURN_IF_ERROR(TensorBase_init(out, aggregated_shape, in->ndim));

    // Loop through each element in the input tensors data, and map what position in the aggregated tensors data.
    scalar temporary_output_buffer[out->numel];
    memset(temporary_output_buffer, 0, sizeof(scalar) * out->numel);

    for (long in_data_index = 0; in_data_index < in->numel; in_data_index++)
    {
        long temporary_index = in_data_index;
        long out_data_index = 0;
        for (long dim = out->ndim - 1; dim >= 0; dim--)
        {
            // Calculate the coordinate within the current dimension of the input tensor.
            long in_coordinate_at_current_dimension = temporary_index % in->shape[dim];

            // Calculate the corresponding coordinate within the current dimension of the output tensor.
            // If dimensions_to_aggregate[dim] is 1, !dimensions_to_aggregate[dim] is 0, and the coordinate is reduced.
            // Vice versa, if 0, it is preserved.
            long out_coordinate_at_current_dimension = !dimensions_to_aggregate[dim] * in_coordinate_at_current_dimension;

            // Update the linear index for the output data by adding the offset caused by the
            // coordinate in the current dimension, scaled by the stride for that dimension.
            // Strides are used to calculate the memory offset based on multi-dimensional indices.
            out_data_index += out_coordinate_at_current_dimension * out->strides[dim];

            // Update the temporary linear index of the input tensor by removing the contribution
            // of the current dimension.
            temporary_index /= in->shape[dim];
        }

        switch (agg)
        {
        case SCALAR_AGG_MEAN:
        case SCALAR_AGG_SUM:
            // For both MEAN and SUM aggregations, accumulate the input data into the temporary output buffer.
            temporary_output_buffer[out_data_index] += in->data[in_data_index];
            break;
        default:
            return INVALID_OPERATION;
        }
    }

    if (agg == SCALAR_AGG_MEAN)
    {
        scalar num_elements_in_aggregation = in->numel / out->numel;
        for (long i = 0; i < out->numel; i++)
        {
            temporary_output_buffer[i] /= num_elements_in_aggregation;
        }
    }

    memcpy(out->data, temporary_output_buffer, out->numel * sizeof(scalar));

    if (keepdim)
    {
        return OK;
    }

    ShapeArray aggregated_shape_without_kept_dimensions;
    long ndim_without_kept_dimensions = 0;
    for (long dim = 0; dim < in->ndim; dim++)
    {
        // If the current dimension is being aggregated, do not include it in the new aggregated shape.
        if (dimensions_to_aggregate[dim] == 0)
        {
            aggregated_shape_without_kept_dimensions[ndim_without_kept_dimensions] = aggregated_shape[dim];
            ndim_without_kept_dimensions++; // Increment the number of dimensions in the new shape
        }
    }
    for (long dim = ndim_without_kept_dimensions; dim < MAX_RANK; dim++)
    {
        aggregated_shape_without_kept_dimensions[dim] = -1;
    }

    // Reshape the output tensor 'out' to the newly calculated aggregated shape.
    RETURN_IF_ERROR(TensorBase_reshape_inplace(out, aggregated_shape_without_kept_dimensions, ndim_without_kept_dimensions));

    return OK;
}