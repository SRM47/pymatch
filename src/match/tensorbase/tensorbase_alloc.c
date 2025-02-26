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

    for (int dim = ndim; dim < MAX_RANK; dim++)
    {
        shape[dim] = -1;
    }

    // Calculate number of elements.
    long numel = 1;
    long numel_for_stride = 1;
    for (int dim = 0; dim < ndim; dim++)
    {
        long dimension_size = shape[dim];
        if (dimension_size < 0)
        {
            return INVALID_DIMENSION_SIZE;
        }
        numel *= dimension_size;
        if (dimension_size > 0)
        {
            numel_for_stride *= dimension_size;
        }
    }

    // Calculate strides.
    StrideArray strides;
    memset(strides, 0, MAX_RANK * sizeof(long));
    long stride = numel_for_stride;
    for (long dim = 0; dim < ndim; dim++)
    {
        long dimension_size = shape[dim];
        if (dimension_size > 0)
        {
            stride /= dimension_size;
        }
        strides[dim] = stride;
    }

    // Allocate (ONLY) the memory for the underlying data of the TensorBase struct.
    // If the TensorBase is singleton, the data pointer will hold the value instead of pointing to a one element array.
    scalar *data;
    if (ndim != 0)
    {
        data = (scalar *)malloc(numel * sizeof(scalar));
        if (data == NULL)
        {
            return MALLOC_ERR;
        }
    }
    else
    {
        // If singleton, initialize the pointer to be null.
        data = NULL;
    }

    td->numel = numel;
    td->ndim = ndim;
    memcpy(td->shape, shape, MAX_RANK * sizeof(long));
    memcpy(td->strides, strides, MAX_RANK * sizeof(long));
    td->data = data;

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