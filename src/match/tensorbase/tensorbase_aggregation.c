#include "tensorbase.h"
#include "tensorbase_util.c"

StatusCode TensorBase_aggregate(TensorBase *in, IndexArray aggregation_dimensions, int keepdim, TensorBase *out, AggScalarOperation agg)
{
    if (in == NULL || out == NULL)
    {
        return TB_NULL_INPUT_ERROR;
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
            return TB_DIMENSION_OUT_OF_BOUNDS_ERROR;
        }
        if (dimensions_to_aggregate[dim] != 0)
        {
            return TB_DUPLICATE_DIMENSION_ERROR;
        }
        dimensions_to_aggregate[dim] = 1;
    }

    if (TensorBase_is_singleton(in))
    {
        memcpy(out, in, sizeof(TensorBase));

        if (agg == SCALAR_AGG_ARGMAX || agg == SCALAR_AGG_ARGMIN)
        {
            // The argmax/argmin of a singleton is the only element in the tensor.
            out->data = 0;
        }

        return TB_OK;
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

    // Boolean to indicate if only one dimension is being reduced (for ARGMAX and ARGMIN).
    // If only one dimension is being reduced (aggregated away), then the second element in aggregation_dimensions would be negative (not set).
    bool only_one_dimensions_reduced = aggregation_dimensions[1] < 0;

    // Initialize the output tensor with the new shape.
    RETURN_IF_ERROR(TensorBase_init(out, aggregated_shape, in->ndim));

    // Store a temporary buffer that holds the output (aggregated) tensors data on the stack for fast access.
    scalar temporary_output_buffer[out->numel];
    // Store another temporary buffer to store the max and min elements for MAX, MIN, ARGMAX, and ARGMIN aggregation operations.
    scalar min_max_buffer[out->numel];

    // Initialization of this array depends on the aggregation operation.
    switch (agg)
    {
    case SCALAR_AGG_MAX:
    case SCALAR_AGG_ARGMAX:
        for (long i = 0; i < out->numel; i++)
        {
            min_max_buffer[i] = -DBL_MAX;
        }
        break;
    case SCALAR_AGG_MIN:
    case SCALAR_AGG_ARGMIN:
        for (long i = 0; i < out->numel; i++)
        {
            min_max_buffer[i] = DBL_MAX;
        }
        break;
    case SCALAR_AGG_SUM:
    case SCALAR_AGG_MEAN:
        memset(temporary_output_buffer, 0, sizeof(scalar) * out->numel);
        break;
    default:
        break;
    }

    // Loop through each element in the input tensors data, and map what position in the aggregated tensors data.
    for (long in_data_index = 0; in_data_index < in->numel; in_data_index++)
    {
        long temporary_index = in_data_index;
        long out_data_index = 0;
        IndexArray in_coordinate;
        // At this point, the dimensions of the input and output tensors are equal.
        for (long dim = out->ndim - 1; dim >= 0; dim--)
        {
            // Calculate the coordinate within the current dimension of the input tensor.
            long in_coordinate_at_current_dimension = temporary_index % in->shape[dim];
            in_coordinate[dim] = in_coordinate_at_current_dimension;

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

        scalar temp = min_max_buffer[out_data_index];;

        switch (agg)
        {
        case SCALAR_AGG_MAX:
            if (in->data[in_data_index] >= temp)
            {
                min_max_buffer[out_data_index] = in->data[in_data_index];
                temporary_output_buffer[out_data_index] = in->data[in_data_index];
            }
            break;
        case SCALAR_AGG_MIN:
            if (in->data[in_data_index] < temp)
            {
                min_max_buffer[out_data_index] = in->data[in_data_index];
                temporary_output_buffer[out_data_index] = in->data[in_data_index];
            }
            break;
        case SCALAR_AGG_ARGMAX:
            if (in->data[in_data_index] >= temp)
            {
                min_max_buffer[out_data_index] = in->data[in_data_index];
                temporary_output_buffer[out_data_index] = only_one_dimensions_reduced ? in_coordinate[aggregation_dimensions[0]] : in_data_index;
            }
            break;
        case SCALAR_AGG_ARGMIN:
            if (in->data[in_data_index] < temp)
            {
                min_max_buffer[out_data_index] = in->data[in_data_index];
                temporary_output_buffer[out_data_index] = only_one_dimensions_reduced ? in_coordinate[aggregation_dimensions[0]] : in_data_index;
            }
            break;
        case SCALAR_AGG_MEAN:
        case SCALAR_AGG_SUM:
            // For both MEAN and SUM aggregations, accumulate the input data into the temporary output buffer.
            temporary_output_buffer[out_data_index] += in->data[in_data_index];
            break;
        default:
            break;
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
        return TB_OK;
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

    return TB_OK;
}