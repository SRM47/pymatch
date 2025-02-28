#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include "tensorbase.h"
#include "tensorbase_util.c"

StatusCode TensorBase_init(TensorBase *tb, ShapeArray shape, long ndim)
{
    if (ndim > MAX_RANK || ndim < 0)
    {
        return NDIM_OUT_OF_BOUNDS;
    }

    if (tb == NULL)
    {
        return NULL_INPUT_ERR;
    }

    for (int dim = ndim; dim < MAX_RANK; dim++)
    {
        shape[dim] = -1;
    }

    // `numel` stores the total number of elements in the tensor.
    long numel = 1;
    // `numel_for_strides` is calculated differently, multiplying only dimensions with sizes greater than zero.
    long numel_for_stride = 1;
    // This distinction is crucial for stride calculation. A dimension of size zero effectively has a stride of 1.
    // To simplify stride computations, we use `numel_for_strides` which treats zero-sized dimensions as contributing a multiplicative factor of 1, rather than 0.
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
    // Strides tell you how many elements you need to skip in memory to move to the next element along a specific dimension.
    // For instance, suppose a tensor with a shape [2,3,4], the strides array will be [12, 4, 1].
    // * The stride of the first dimension (2), is the number of elements (in memory) between [0,0,0] and [1,0,0], which is 12.
    // * The stride of the second dimension (3), is the number of elements (in memory) between [0,1,0] and [0,1,0], which is 4.
    // * The stride of the third (last) dimension (4), is the number of elements (in memory) between [0,0,0] and [0,0,1], which is 1.
    // Computationally, the stride at a specific dimension is the product of all latter dimensions.
    StrideArray strides;
    memset(strides, 0, MAX_RANK * sizeof(long)); // Initialize strides to 0.
    long stride = numel_for_stride;              // Start with total element count (excluding zero-sized dims).
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

    tb->numel = numel;
    tb->ndim = ndim;
    memcpy(tb->shape, shape, MAX_RANK * sizeof(long));
    memcpy(tb->strides, strides, MAX_RANK * sizeof(long));
    tb->data = data;

    return OK;
}

void TensorBase_dealloc(TensorBase *tb)
{
    if (tb == NULL)
    {
        return;
    }

    // Frees tensorbase data, except for singletons. Assumes exclusive data ownership to avoid implementing reference counting.
    // Singletons store data directly, not on the heap (i.e., with malloc).
    if (tb->data != NULL && !TensorBase_is_singleton(tb))
    {
        free(tb->data);
    }
    // Use memset to zero out shape and strides arrays safely.
    memset(tb, 0, sizeof(TensorBase));
}