#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include "tensorbase.h"
#include "tensorbase_util.c"

typedef struct
{
    scalar a;
    scalar b;
} randn_pair;

// Box-Muller method for generating normally distributed random numbers.
// https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform#C++
static randn_pair randn(scalar mu, scalar sigma)
{
    scalar two_pi = 2.0 * M_PI;

    scalar u1;
    do
    {
        u1 = (scalar)rand() / (scalar)((unsigned)RAND_MAX + 1);
    } while (u1 == 0);

    scalar u2 = (scalar)rand() / (scalar)((unsigned)RAND_MAX + 1);

    scalar mag = sigma * sqrt(-2.0 * log(u1));

    randn_pair result = {0};
    result.a = mag * cos(two_pi * u2) + mu;
    result.b = mag * sin(two_pi * u2) + mu;

    return result;
}

int TensorBase_permute(TensorBase *in, IndexArray permutation, TensorBase *out)
{
    if (in == NULL || out == NULL)
    {
        return -1;
    }

    if (TensorBase_is_singleton(in))
    {
        if (permutation[0] >= 0)
        {
            return -1;
        }
        return TensorBase_create_empty_like(in, out);
    }

    // Check if permutation is a valid permutation
    ShapeArray permuted_shape;
    IndexArray seen_dimensions;
    memset(seen_dimensions, 0, sizeof(scalar) * MAX_RANK);
    long dim = 0;
    for (; dim < in->ndim; dim++)
    {
        if (permutation[dim] < 0)
        {
            return -1;
        }
        if (permutation[dim] >= in->ndim)
        {
            return -1;
        }
        if (seen_dimensions[permutation[dim]] != 0)
        {
            return -1; // Duplicate
        }
        seen_dimensions[permutation[dim]] = 1;
        permuted_shape[dim] = in->shape[permutation[dim]];
    }

    RETURN_IF_ERROR(TensorBase_init(out, permuted_shape, in->ndim));

    for (long in_data_index = 0; in_data_index < in->numel; in_data_index++)
    {
        // For each element in the data index, calculate the corresponding
        // element in each of the input tensors.
        long out_data_index = 0;

        IndexArray in_coord;
        for (long dim = in->ndim - 1, temp = in_data_index; dim >= 0; dim--)
        {
            in_coord[dim] = temp % in->shape[dim];
            temp /= in->shape[dim];
        }

        for (long out_dim = 0; out_dim < out->ndim; out_dim++)
        {
            out_data_index += out->strides[out_dim] * in_coord[permutation[out_dim]];
        }

        out->data[out_data_index] = in->data[in_data_index];
    }

    return 0;
}

int TensorBase_reshape_inplace(TensorBase *in, ShapeArray shape, long ndim)
{
    // Does not do validation of shape array
    // assumes ndim is the rank of shape
    if (in == NULL)
    {
        return -1;
    }

    if (ndim > MAX_RANK)
    {
        return -4;
    }

    long numel_for_stride = 1;
    long numel = 1;
    printf("%ld ", ndim);
    for (int i = 0; i < ndim; i++)
    {
        long dim = shape[i];
        printf("%ld ", dim);
        if (dim < 0)
        {
            // All dimensions must be >= 0 (-1 indicates no-dimension).
            return -2;
        }
        numel *= dim;
        if (dim > 0)
        {
            numel_for_stride *= dim;
        }
    }

    if (numel != in->numel)
    {
        return -3;
    }

    // Calculate if the input tensor is singleton before changing internal metadata.
    int is_singleton = TensorBase_is_singleton(in);

    // Calculate Tensorbase strides.
    in->ndim = ndim;
    long stride = numel_for_stride;
    long i = 0;
    for (; i < ndim; i++)
    {
        long dim = shape[i];
        if (dim > 0)
        {
            stride /= dim;
        }
        in->strides[i] = stride;
        in->shape[i] = dim;
    }
    for (; i < MAX_RANK; i++)
    {
        in->strides[i] = 0;
        in->shape[i] = -1;
    }

    // Must handle special case for singleton objects.
    // Singleton -> nd. must allocate memory for the one element in the nd array
    if (is_singleton)
    {
        if (ndim > 0)
        {
            // Allocate the new memory.
            scalar *new_data_region = (scalar *)malloc(1 * sizeof(scalar));
            if (new_data_region == NULL)
            {
                return -1;
            }
            // Copy the bits of the original scalar into the first position of the new_data_region.
            memcpy(new_data_region, &(in->data), sizeof(scalar));
            // Assign the data point the new region in memory.
            in->data = new_data_region;
        }
    }
    // nd -> singleton. must free memory and place the singleton value in the pointer.
    else
    {
        // By this point, if ndim is 0, the bew shape will be a singleton and all the checks wouldv'e been verified.
        // this is just copying the data.
        if (ndim == 0)
        {
            // also note that the fact that the singleton data isnt just a one element array meas that it's hard now to share data
            // we lose that functionality for better locality and fewer allocations.
            // Get the single value from the nd tensor
            scalar value = *in->data;
            // Free the memory
            free(in->data);
            // Copy the memory bits from the scalar into the in->data pointer
            memcpy(&(in->data), &value, sizeof(scalar));
        }
    }

    return 0;
}
int TensorBase_reshape(TensorBase *in, TensorBase *out, ShapeArray shape, long ndim)
{
    // Does not do validation of shape array
    // assumes ndim is the rank of shape
    if (in == NULL || out == NULL)
    {
        return -1;
    }

    if (ndim > MAX_RANK)
    {
        return -4;
    }

    // will not share data memory (data pointers point to the same data) or else we'd have to implement reference counting of the data element
    // well call this a limitation of the system
    RETURN_IF_ERROR(TensorBase_create_empty_like(in, out));

    // must also copy the data over if not a single object.
    if (!TensorBase_is_singleton(in))
    {
        memcpy(out->data, in->data, out->numel * sizeof(scalar));
    }

    return TensorBase_reshape_inplace(out, shape, ndim);
}

int TensorBase_fill_(TensorBase *in, scalar fill_value)
{
    if (TensorBase_is_singleton(in))
    {
        memcpy(&(in->data), &fill_value, sizeof(scalar));
        return 0;
    }

    if (in->data == NULL)
    {
        return -1;
    }

    // Don't use memset for doubles/floats. Only for chars.
    for (long i = 0; i < in->numel; i++)
    {
        in->data[i] = fill_value;
    }

    return 0;
}
int TensorBase_randn_(TensorBase *in, scalar mu, scalar sigma)
{
    // Assumes tensor is already initialized. Just filling the data with random, normallu distributed values.
    if (TensorBase_is_singleton(in))
    {
        randn_pair pair = randn(mu, sigma);
        memcpy(&in->data, &pair.a, sizeof(scalar));
        return 0;
    }

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

    return 0;
}

int TensorBase_item(TensorBase *t, scalar *item)
{
    if (t->numel != 1)
    {
        return -1;
    }

    if (TensorBase_is_singleton(t))
    {
        memcpy(item, &(t->data), sizeof(scalar));
    }
    else
    {
        *item = *(t->data);
    }

    return 0;
}