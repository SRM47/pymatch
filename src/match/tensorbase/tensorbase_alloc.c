#include "tensorbase.h"
#include "tensorbase_util.c"

StatusCode TensorBase_init(TensorBase *tb, ShapeArray shape, long ndim)
{
    if (ndim > MAX_RANK || ndim < 0)
    {
        return TB_INVALID_NDIM_ERROR;
    }

    if (tb == NULL)
    {
        return TB_NULL_INPUT_ERROR;
    }

    for (int dim = ndim; dim < MAX_RANK; dim++)
    {
        shape[dim] = -1;
    }

    // `numel` stores the total number of elements in the tensor.
    long numel = 1;
    for (long dim = 0; dim < ndim; dim++)
    {
        if (shape[dim] < 0)
        {
            return TB_INVALID_DIMENSION_SIZE_ERROR;
        }
        numel *= shape[dim];
    }

    // Calculate Strides of the tensor.
    StrideArray strides;
    RETURN_IF_ERROR(calculate_strides_from_shape(shape, ndim, strides));

    // Allocate (ONLY) the memory for the underlying data of the TensorBase struct.
    // If the TensorBase is singleton, the data pointer will hold the value instead of pointing to a one element array.
    scalar *data;
    if (ndim != 0)
    {
        data = (scalar *)malloc(numel * sizeof(scalar));
        if (data == NULL)
        {
            return TB_MALLOC_ERROR;
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

    return TB_OK;
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