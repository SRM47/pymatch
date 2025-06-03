#include "tensorbase.h"
#include "tensorbase_util.c"

StatusCode TensorBase_permute(TensorBase *in, IndexArray permutation, long ndim, TensorBase *out)
{
    if (in == NULL || out == NULL)
    {
        return NULL_INPUT_ERR;
    }

    if (in->ndim != ndim)
    {
        return PERMUTATION_INCORRECT_NDIM;
    }

    if (TensorBase_is_singleton(in))
    {
        return TensorBase_create_empty_like(in, out);
    }

    // Check if permutation is a valid permutation
    ShapeArray permuted_shape;
    IndexArray seen_dimensions;
    memset(seen_dimensions, 0, sizeof(scalar) * MAX_RANK);

    for (long i = 0; i < in->ndim; i++)
    {
        long dim = permutation[i];
        if (dim < 0 || dim >= in->ndim)
        {
            return INVALID_DIMENSION;
        }
        if (seen_dimensions[dim] != 0)
        {
            return PERMUTATION_DUPLICATE_DIM;
        }
        seen_dimensions[dim] = 1;
        permuted_shape[i] = in->shape[dim];
    }

    RETURN_IF_ERROR(TensorBase_init(out, permuted_shape, ndim));

    for (long in_data_index = 0; in_data_index < in->numel; in_data_index++)
    {
        // Map each input element to its output location, applying the permutation.
        // Example: [3,4,5] with permutation (1,0,2) swaps the first two dimensions, e.g., (x,y,z) -> (y,x,z).

        // Decompose the flat `in_data_index` into its corresponding multi-dimensional index into the `in` tensor.
        IndexArray in_coord;
        for (long dim = in->ndim - 1, temp = in_data_index; dim >= 0; dim--)
        {
            in_coord[dim] = temp % in->shape[dim];
            temp /= in->shape[dim];
        }

        // Shift the dimensions of the input coordinate to obtain the permuted coordinate in the out tensor.
        IndexArray out_coord;
        for (long out_dim = 0; out_dim < out->ndim; out_dim++)
        {
            out_coord[out_dim] = in_coord[permutation[out_dim]];
        }

        long out_data_index = 0;
        for (long out_dim = 0; out_dim < out->ndim; out_dim++)
        {
            out_data_index += out->strides[out_dim] * out_coord[out_dim];
        }

        // Or...just this. I leave this to the reader to figure out how this works.
        // long out_data_index = 0;
        // for (long out_dim = 0; out_dim < out->ndim; out_dim++)
        // {
        //     out_data_index += out->strides[out_dim] * in_coord[permutation[out_dim]];
        // }

        out->data[out_data_index] = in->data[in_data_index];
    }

    return OK;
}

StatusCode TensorBase_transpose(TensorBase *in, TensorBase *out)
{
    if (in == NULL || out == NULL)
    {
        return NULL_INPUT_ERR;
    }

    long ndim = in->ndim;

    IndexArray permutation;
    memset(permutation, 0, MAX_RANK * sizeof(long));
    for (long i = 0; i < ndim; i++)
    {
        permutation[i] = ndim-i-1;
    }

    return TensorBase_permute(in, permutation, ndim, out);
}

StatusCode TensorBase_reshape_inplace(TensorBase *in, ShapeArray shape, long ndim)
{
    if (in == NULL)
    {
        return NULL_INPUT_ERR;
    }

    if (ndim > MAX_RANK)
    {
        return NDIM_OUT_OF_BOUNDS;
    }

    // Verify that the new shape is valid by comparing the number of elements in the existing and new shapes.
    long temp_numel = 1;
    for (long dim = 0; dim < ndim; dim++)
    {
        if (shape[dim] < 0)
        {
            return INVALID_DIMENSION_SIZE;
        }
        temp_numel *= shape[dim];
    }
    if (temp_numel != in->numel)
    {
        return RESHAPE_INVALID_SHAPE_NUMEL_MISMATCH;
    }

    // Calculate Strides of the tensor.
    StrideArray strides;
    RETURN_IF_ERROR(calculate_strides_from_shape(shape, ndim, strides));

    // There is a special case for singleton tensors.
    // If a singleton is reshaped into an n-dimensional tensor with a single element,
    // we must allocate memory on the heap for a single element.
    if (TensorBase_is_singleton(in))
    {
        if (ndim > 0)
        {
            // Allocate the new memory.
            scalar *new_data_region = (scalar *)malloc(1 * sizeof(scalar));
            if (new_data_region == NULL)
            {
                return MALLOC_ERR;
            }
            // Copy the bits of the original scalar into the first position of the new_data_region.
            memcpy(new_data_region, &(in->data), sizeof(scalar));
            // Assign the data point the new region in memory.
            in->data = new_data_region;
        }
    }
    else
    {
        // If an n-dimensional tensor with a single element was reshaped into a singleton,
        // we must deallocate memory on the heap and store the original element in the pointer value.
        if (ndim == 0)
        {
            // Get the single value from the n-dimensional tensor.
            scalar value = *in->data;
            // Free the memory.
            free(in->data);
            // Copy the memory bits from the scalar into the in->data pointer/
            memcpy(&(in->data), &value, sizeof(scalar));
        }
    }

    // Update the Shape, Strides and ndim inplace.
    memcpy(in->strides, strides, MAX_RANK * sizeof(long));
    memcpy(in->shape, shape, MAX_RANK * sizeof(long));
    in->ndim = ndim;

    return OK;
}

StatusCode TensorBase_reshape(TensorBase *in, TensorBase *out, ShapeArray shape, long ndim)
{
    if (in == NULL || out == NULL)
    {
        return NULL_INPUT_ERR;
    }

    if (ndim > MAX_RANK)
    {
        return NDIM_OUT_OF_BOUNDS;
    }

    RETURN_IF_ERROR(TensorBase_create_empty_like(in, out));

    // must also copy the data over if not a single object.
    if (!TensorBase_is_singleton(in))
    {
        memcpy(out->data, in->data, out->numel * sizeof(scalar));
    }

    return TensorBase_reshape_inplace(out, shape, ndim);
}

StatusCode TensorBase_fill_(TensorBase *in, scalar fill_value)
{
    if (in == NULL)
    {
        return NULL_INPUT_ERR;
    }

    if (TensorBase_is_singleton(in))
    {
        memcpy(&(in->data), &fill_value, sizeof(scalar));
        return OK;
    }

    if (in->data == NULL)
    {
        return NULL_INPUT_ERR;
    }

    for (long i = 0; i < in->numel; i++)
    {
        in->data[i] = fill_value;
    }

    return OK;
}

StatusCode TensorBase_randn_(TensorBase *in, scalar mu, scalar sigma)
{
    if (in == NULL)
    {
        return NULL_INPUT_ERR;
    }

    if (TensorBase_is_singleton(in))
    {
        randn_pair pair = randn(mu, sigma);
        memcpy(&in->data, &pair.a, sizeof(scalar));
        return OK;
    }
    // Assumes tensor is already initialized with a valid `data` pointer.
    scalar *data = in->data;
    for (long index = 0; index < in->numel; index += 2)
    {
        randn_pair pair = randn(mu, sigma);
        data[index] = pair.a;
        if (index + 1 < in->numel)
        {
            data[index + 1] = pair.b;
        }
    }

    return OK;
}

StatusCode TensorBase_item(TensorBase *t, scalar *item)
{
    if (t->numel != 1)
    {
        return ITEM_NUMEL_NOT_ONE;
    }

    if (TensorBase_is_singleton(t))
    {
        memcpy(item, &(t->data), sizeof(scalar));
    }
    else
    {
        *item = *(t->data);
    }

    return OK;
}