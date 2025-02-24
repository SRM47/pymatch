#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include "tensorbase.h"
#include "tensorbase_util.c"

// Assumes that index is already a valid coordinate!
StatusCode calculate_shape_from_subscrtips(SubscriptArray subscripts,
                                           long num_subscripts,
                                           ShapeArray tb_shape,
                                           ShapeArray shape,
                                           long *ndim)
{
    // Validate input parameters
    if (!subscripts || !shape || !ndim || num_subscripts > MAX_RANK)
    {
        return NULL_INPUT_ERR;
    }

    long new_ndim = 0;

    // Process each subscript
    for (long i = 0; i < num_subscripts; i++)
    {
        TensorBaseSubscript *sub = subscripts + i;
        long slice_start = sub->start;
        long slice_stop = sub->stop;
        long slice_step = sub->step;
        long slice_size;

        switch (sub->type)
        {
        case INDEX:
            // For index access, dimension is removed
            // Validate the index is within bounds
            if (slice_start < 0 || slice_start >= tb_shape[i])
            {
                return INDEX_OUT_OF_BOUNDS;
            }
            // Don't increment new_ndim as this dimension is eliminated
            break;

        case SLICE:
            // For slice access, calculate the new dimension size

            // Validate slice parameters
            if (slice_start < 0 || slice_stop < 0 || slice_step <= 0 ||
                slice_start >= tb_shape[i])
            {
                return STATUS_SUBSCRIPT_INVALID_PARAMETER;
            }

            slice_stop = min_long(slice_stop, tb_shape[i]);

            // Calculate the size of the resulting dimension
            slice_size = (slice_stop - slice_start + slice_step - 1) / slice_step;

            // Store the new dimension size
            if (new_ndim >= MAX_RANK)
            {
                return NDIM_OUT_OF_BOUNDS;
            }
            shape[new_ndim] = slice_size;
            new_ndim++;
            sub->stop = slice_stop;
            break;
        default:
            return NULL_INPUT_ERR;
        }
    }

    // Update the number of dimensions
    *ndim = new_ndim;

    // Fill in the rest of the shape array
    for (long i = new_ndim; i < MAX_RANK; i++)
    {
        shape[i] = -1;
    }

    return OK;
}

void get_next_coordinate(SubscriptArray subscripts, long num_subscrtips, IndexArray coord)
{
    long dim = num_subscrtips - 1;
    // Find the right-most slice component to the key
    while (subscripts[dim].type != SLICE)
    {
        dim--;
    }

    // Increment the coordinates dimension value by the step size of the slice.
    coord[dim] += subscripts[dim].step;
    // If the new value is less than the stop value for that slice, return as normal.
    // If the new value is greater than or equal to the stop value for that slice, recursively increase the previous dimensions by the step slice.
    while ((coord[dim] >= subscripts[dim].stop || subscripts[dim].type != SLICE) && dim >= 0)
    {
        // Reset the current dimension back to the start
        coord[dim] = subscripts[dim].start;
        // Inspect the previous dimension
        dim--;
        if (dim < 0)
        {
            return;
        }
        // Increase that dimension by its step size.
        coord[dim] += subscripts[dim].step;
    }
}

// Returns an array of all of the indices in a TensorBase object to pull data from based on the subscripts.
StatusCode TensorBase_get(TensorBase *in, SubscriptArray subscripts, long num_subscripts, TensorBase *out)
{
    // Calculate the resulting shape of the output array and initialize new data array with malloc
    long new_ndim;
    ShapeArray new_shape;
    RETURN_IF_ERROR(calculate_shape_from_subscrtips(subscripts, num_subscripts, in->shape, new_shape, &new_ndim));

    print_long_list(new_shape, 8);

    // Initialize the out tensorbase
    RETURN_IF_ERROR(TensorBase_init(out, new_shape, new_ndim));

    IndexArray curr_index;
    memset(curr_index, 0, MAX_RANK * sizeof(long));
    // Initialize curr_index to the starting position of all subscripts.
    for (long i = 0; i < num_subscripts; i++)
    {
        curr_index[i] = subscripts[i].start;
    }
    print_long_list(curr_index, 8);

    if (TensorBase_is_singleton(out))
    {
        long in_data_index = 0;
        RETURN_IF_ERROR(TensorBase_convert_indices_to_data_index(in, curr_index, &in_data_index));
        memcpy(&out->data, in->data + in_data_index, sizeof(scalar));
        return OK;
    }

    // Start increasing here and go down
    long curr_dim = num_subscripts - 1;
    long new_numel = out->numel;
    scalar *new_data = out->data;
    for (long out_data_index = 0; out_data_index < new_numel; out_data_index++)
    {
        // Convert index into data indices
        long in_data_index;
        RETURN_IF_ERROR(TensorBase_convert_indices_to_data_index(in, curr_index, &in_data_index));
        new_data[out_data_index] = in->data[in_data_index];

        // Update the curr_index Index array.
        get_next_coordinate(subscripts, num_subscripts, curr_index);
    }

    return OK;
}

StatusCode TensorBase_set_scalar(TensorBase *in, SubscriptArray subscripts, long num_subscripts, scalar s)
{
    // Calculate the resulting shape of the output array and initialize new data array with malloc
    long new_ndim;
    ShapeArray new_shape;
    RETURN_IF_ERROR(calculate_shape_from_subscrtips(subscripts, num_subscripts, in->shape, new_shape, &new_ndim));

    long numel_to_set = 1;
    for (long i = 0; i < new_ndim; i++)
    {
        numel_to_set *= new_shape[i];
    }

    if (TensorBase_is_singleton(in))
    {
        memcpy(&in->data, &s, sizeof(scalar));
        return OK;
    }

    IndexArray curr_index;
    memset(curr_index, 0, MAX_RANK * sizeof(long));
    // Initialize curr_index to the starting position of all subscripts.
    for (long i = 0; i < num_subscripts; i++)
    {
        curr_index[i] = subscripts[i].start;
    }
    print_long_list(curr_index, 8);

    // Start increasing here and go down
    long curr_dim = num_subscripts - 1;
    scalar *data = in->data;
    for (long i = 0; i < numel_to_set; i++)
    {
        // Convert index into data indices
        long in_data_index;
        RETURN_IF_ERROR(TensorBase_convert_indices_to_data_index(in, curr_index, &in_data_index));
        data[in_data_index] = s;

        // Update the curr_index Index array.
        get_next_coordinate(subscripts, num_subscripts, curr_index);
    }

    return OK;
}

StatusCode TensorBase_set_tensorbase(TensorBase *in, SubscriptArray subscripts, long num_subscripts, TensorBase *t)
{
    // Calculate the resulting shape of the output array and initialize new data array with malloc
    long new_ndim;
    ShapeArray new_shape;
    RETURN_IF_ERROR(calculate_shape_from_subscrtips(subscripts, num_subscripts, in->shape, new_shape, &new_ndim));

    // Verify that the shape from the subscripts (key) is the same as the tensorbase to set
    if (!TensorBase_same_shape(new_shape, t->shape))
    {
        return INVALID_SHAPES_FOR_SET;
    }

    long numel_to_set = t->numel;
    IndexArray curr_index;
    memset(curr_index, 0, MAX_RANK * sizeof(long));
    // Initialize curr_index to the starting position of all subscripts.
    for (long i = 0; i < num_subscripts; i++)
    {
        curr_index[i] = subscripts[i].start;
    }
    print_long_list(curr_index, 8);

    if (TensorBase_is_singleton(t))
    {
        long in_data_index = 0;
        RETURN_IF_ERROR(TensorBase_convert_indices_to_data_index(in, curr_index, &in_data_index));
        memcpy(in->data + in_data_index, &t->data, sizeof(scalar));
        return OK;
    }

    // Start increasing here and go down
    long curr_dim = num_subscripts - 1;
    scalar *in_data = in->data;
    scalar *out_data = t->data;
    for (long i = 0; i < numel_to_set; i++)
    {
        // Convert index into data indices
        long in_data_index;
        RETURN_IF_ERROR(TensorBase_convert_indices_to_data_index(in, curr_index, &in_data_index));
        in_data[in_data_index] = out_data[i];

        // Update the curr_index Index array.
        get_next_coordinate(subscripts, num_subscripts, curr_index);
    }

    return OK;
}