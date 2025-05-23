#include "tensorbase.h"
#include "tensorbase_util.c"

static StatusCode process_subscripts_for_indexing(SubscriptArray subscripts,
                                                  long num_subscripts,
                                                  ShapeArray original_shape)
{
    if (!subscripts || !original_shape)
    {
        return TB_NULL_INPUT_ERROR;
    }
    if (num_subscripts > MAX_RANK)
    {
        return TB_INVALID_NDIM_ERROR;
    }

    long temporary_sub_ndim = 0;
    for (long dim = 0; dim < num_subscripts; dim++)
    {
        TensorBaseSubscript sub = subscripts[dim];
        long slice_start = sub.start;
        long slice_stop = sub.stop;
        long slice_step = sub.step;
        switch (sub.type)
        {
        case INDEX:
            if (slice_start < 0 || slice_start >= original_shape[dim])
            {
                return TB_INDEX_OUT_OF_BOUNDS_ERROR;
            }
            break;
        case SLICE:
            if (slice_start < 0 || slice_stop < 0 || slice_step <= 0 ||
                slice_start >= original_shape[dim])
            {
                return TB_INDEX_OUT_OF_BOUNDS_ERROR;
            }
            break;
        default:
            return TB_NULL_INPUT_ERROR;
        }
    }

    // Pad the subscript list with full slices to match the tensor's original dimensionality.
    // Ensuring the length of the key equals the tensors dimensionality makes the `get` and `set` methods easier to implement.
    for (long dim = num_subscripts; original_shape[dim] >= 0; dim++)
    {
        subscripts[dim] = (TensorBaseSubscript){SLICE, 0, original_shape[dim], 1};
    }

    return TB_OK;
}

static StatusCode calculate_shape_from_subscrtips(SubscriptArray subscripts,
                                                  long original_ndim,
                                                  ShapeArray original_shape,
                                                  ShapeArray sub_shape,
                                                  long *sub_ndim)
{
    // Assumes that the number of subscripts is equal to the original_ndim. One subscript for every dimension in the input tensor.
    // Call `process_subscripts_for_indexing` prior to this method to ensure this is the case.

    if (!subscripts || !original_shape || !sub_ndim)
    {
        return TB_NULL_INPUT_ERROR;
    }

    long temporary_sub_ndim = 0;
    for (long dim = 0; dim < original_ndim; dim++)
    {
        TensorBaseSubscript *sub = subscripts + dim;
        long slice_start = sub->start;
        long slice_stop = sub->stop;
        long slice_step = sub->step;
        long slice_size;

        switch (sub->type)
        {
        case INDEX:
            // For index access, dimension is removed. So, a new dimension isn't added to the new shape, and new_ndim isn't incremented.
            break;
        case SLICE:
            // For slice access, add the correspoding dimension size to the shape at the current dimension, and incrment new_ndim.
            // Clip the `stop` field to the size of the dimension.
            slice_stop = min_long(slice_stop, original_shape[dim]);
            sub->stop = slice_stop;
            // Calculate the size of the new dimension.
            slice_size = (slice_stop - slice_start + slice_step - 1) / slice_step;
            sub_shape[temporary_sub_ndim] = slice_size;
            temporary_sub_ndim++;
            break;
        default:
            return TB_NULL_INPUT_ERROR;
        }
    }

    *sub_ndim = temporary_sub_ndim;

    for (long i = temporary_sub_ndim; i < MAX_RANK; i++)
    {
        sub_shape[i] = -1;
    }

    return TB_OK;
}

static void get_next_coordinate(SubscriptArray subscripts, long num_subscrtips, IndexArray coord)
{
    long dim = num_subscrtips - 1;
    // Find the right-most slice component to the key.
    while (subscripts[dim].type != SLICE)
    {
        dim--;
    }

    // Increment the coordinates dimension value by the step size of the slice.
    coord[dim] += subscripts[dim].step;
    // If the new value is less than the stop value for that slice, return as normal.
    // If the new value is greater than or equal to the stop value for that slice, recursively increase the previous dimensions by the step slice.
    while (dim >= 0 && coord[dim] >= subscripts[dim].stop)
    {
        // If the current subscript type is an index, skip updating.
        if (subscripts[dim].type == INDEX) {
            dim--;
            continue;
        }
        // Reset the current dimension back to the start.
        coord[dim] = subscripts[dim].start;
        // Inspect the previous dimension.
        dim--;
        if (dim < 0)
        {
            return;
        }
        // Increase that dimension by its step size.
        coord[dim] += subscripts[dim].step;
    }
}

StatusCode TensorBase_get(TensorBase *in, SubscriptArray subscripts, long num_subscripts, TensorBase *subtensor)
{
    long subtensor_ndim;
    ShapeArray subtensor_shape;
    RETURN_IF_ERROR(process_subscripts_for_indexing(subscripts, num_subscripts, in->shape));
    RETURN_IF_ERROR(calculate_shape_from_subscrtips(subscripts, in->ndim, in->shape, subtensor_shape, &subtensor_ndim));
    RETURN_IF_ERROR(TensorBase_init(subtensor, subtensor_shape, subtensor_ndim));

    num_subscripts = in->ndim;

    IndexArray curr_index;
    memset(curr_index, 0, MAX_RANK * sizeof(long));
    for (long i = 0; i < in->ndim; i++)
    {
        curr_index[i] = subscripts[i].start;
    }

    if (TensorBase_is_singleton(subtensor))
    {
        long in_data_index;
        RETURN_IF_ERROR(TensorBase_convert_indices_to_data_index(in, curr_index, &in_data_index));
        memcpy(&subtensor->data, in->data + in_data_index, sizeof(scalar));
        return TB_OK;
    }

    long curr_dim = in->ndim - 1;
    long subtensor_numel = subtensor->numel;
    scalar *subtensor_data = subtensor->data;
    for (long subtensor_data_index = 0; subtensor_data_index < subtensor_numel; subtensor_data_index++)
    {
        long in_data_index;
        RETURN_IF_ERROR(TensorBase_convert_indices_to_data_index(in, curr_index, &in_data_index));
        subtensor_data[subtensor_data_index] = in->data[in_data_index];
        get_next_coordinate(subscripts, num_subscripts, curr_index);
    }

    return TB_OK;
}

StatusCode TensorBase_set_scalar(TensorBase *in, SubscriptArray subscripts, long num_subscripts, scalar s)
{
    if (in == NULL)
    {
        return TB_NULL_INPUT_ERROR;
    }

    if (TensorBase_is_singleton(in))
    {
        memcpy(&in->data, &s, sizeof(scalar));
        return TB_OK;
    }

    long subtensor_ndim;
    ShapeArray subtensor_shape;
    RETURN_IF_ERROR(process_subscripts_for_indexing(subscripts, num_subscripts, in->shape));
    RETURN_IF_ERROR(calculate_shape_from_subscrtips(subscripts, in->ndim, in->shape, subtensor_shape, &subtensor_ndim));

    num_subscripts = in->ndim;

    long subtensor_numel = 1;
    for (long dim = 0; dim < subtensor_ndim; dim++)
    {
        subtensor_numel *= subtensor_shape[dim];
    }

    IndexArray curr_index;
    memset(curr_index, 0, MAX_RANK * sizeof(long));
    for (long i = 0; i < in->ndim; i++)
    {
        curr_index[i] = subscripts[i].start;
    }

    long curr_dim = num_subscripts - 1;
    scalar *data = in->data;
    for (long i = 0; i < subtensor_numel; i++)
    {
        long in_data_index;
        RETURN_IF_ERROR(TensorBase_convert_indices_to_data_index(in, curr_index, &in_data_index));
        data[in_data_index] = s;
        get_next_coordinate(subscripts, num_subscripts, curr_index);
    }

    return TB_OK;
}

StatusCode TensorBase_set_tensorbase(TensorBase *in, SubscriptArray subscripts, long num_subscripts, TensorBase *subtensor)
{
    long calculated_subtensor_ndim_from_subscripts;
    ShapeArray calculated_subtensor_shape_from_subscripts;
    RETURN_IF_ERROR(process_subscripts_for_indexing(subscripts, num_subscripts, in->shape));
    RETURN_IF_ERROR(calculate_shape_from_subscrtips(subscripts, in->ndim, in->shape, calculated_subtensor_shape_from_subscripts, &calculated_subtensor_ndim_from_subscripts));

    if (!TensorBase_same_shape(calculated_subtensor_shape_from_subscripts, subtensor->shape))
    {
        return TB_SHAPE_MISMATCH_ERROR;
    }

    num_subscripts = in->ndim;
    long subtensor_numel = subtensor->numel;

    IndexArray curr_index;
    memset(curr_index, 0, MAX_RANK * sizeof(long));
    for (long i = 0; i < in->ndim; i++)
    {
        curr_index[i] = subscripts[i].start;
    }

    if (TensorBase_is_singleton(subtensor))
    {
        long in_data_index;
        RETURN_IF_ERROR(TensorBase_convert_indices_to_data_index(in, curr_index, &in_data_index));
        memcpy(in->data + in_data_index, &subtensor->data, sizeof(scalar));
        return TB_OK;
    }

    long curr_dim = num_subscripts - 1;
    scalar *in_data = in->data;
    scalar *subtensor_data = subtensor->data;
    for (long subtensor_data_index = 0; subtensor_data_index < subtensor_numel; subtensor_data_index++)
    {
        long in_data_index;
        RETURN_IF_ERROR(TensorBase_convert_indices_to_data_index(in, curr_index, &in_data_index));
        in_data[in_data_index] = subtensor_data[subtensor_data_index];
        get_next_coordinate(subscripts, num_subscripts, curr_index);
    }

    return TB_OK;
}