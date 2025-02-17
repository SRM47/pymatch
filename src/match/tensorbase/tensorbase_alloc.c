#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include "tensorbase.h"
#include "tensorbase_util.c"

StatusCode TensorBase_init(TensorBase *td, ShapeArray shape, long ndim)
{
    
    if (ndim > MAX_RANK || ndim < 0)
    {
        return NDIM_OUT_OF_BOUNDS;
    }

    if (td == NULL)
    {
        return NULL_INPUT_ERR;
    }

    // Initialize Tensorbase instance variables.
    td->numel = 1;
    td->ndim = ndim;
    for (size_t i = 0; i < MAX_RANK; i++)
    {
        td->shape[i] = -1.0;
    }
    memset(td->strides, 0, MAX_RANK * sizeof(long));

    // Calculate Tensorbase shape and number of elements.
    long numel_for_stride = 1;
    for (int i = 0; i < ndim; i++)
    {
        long dim = shape[i];
        if (dim < 0)
        {
            // All dimensions must be >= 0 (-1 indicates no-dimension).
            return INVALID_DIMENSION_SIZE;
        }
        td->numel *= dim;
        td->shape[i] = dim;
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
        td->strides[i] = stride;
    }

    // Allocate the memory, but do not initialize it's values.
    // If TensorBase is singleton, the data pointer will hold the value
    // instead of pointing to a one element array.
    if (ndim != 0)
    {
        td->data = (scalar *)malloc(td->numel * sizeof(scalar));
        if (td->data == NULL)
        {
            // Memory error.
            return MALLOC_ERR;
        }
    }
    else
    {
        // If singleton, initialize the pointer to be null.
        // TensorBase_create_tensor_like will not free this then,
        td->data = NULL;
    }

    return OK;
}

void TensorBase_dealloc(TensorBase *td)
{
    if (td == NULL)
    {
        return;
    }

    // Only free the pointer if not singleton.
    // Sington tensor structs do not point to address on heap,
    // rather directly store data in the pointer variable.
    // also assumes only one pointer to data. will not implementreference counting if we start to share memory between tensors
    if (td->data != NULL && !TensorBase_is_singleton(td))
    {
        free(td->data);
    }
    // Use memset to zero out shape and strides arrays safely.
    memset(td, 0, sizeof(TensorBase));
}