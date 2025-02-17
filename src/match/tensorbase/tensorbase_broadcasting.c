#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include "tensorbase.h"
#include "tensorbase_util.c"

static int TensorBase_can_broadcast(TensorBase *in, ShapeArray broadcast_shape, int broadcast_ndim)
{
    if (in->ndim > broadcast_ndim)
    {
        return -1;
    }

    // Broadcast dimensions
    long in_index = in->ndim - 1;
    long out_index = broadcast_ndim - 1;

    for (; in_index >= 0; out_index--, in_index--)
    {
        if (in->shape[in_index] == 1 || broadcast_shape[out_index] == 1 || in->shape[in_index] == broadcast_shape[out_index])
        {
            continue;
        }
        return -1;
    }

    return 0;
}
// TODO: Implement
int TensorBase_broadcast_to(TensorBase *in, ShapeArray broadcast_shape, int *broadcast_ndim, TensorBase *out)
{
    return -2;
}

int TensorBase_unbroadcast(TensorBase *in, ShapeArray shape, long ndim, TensorBase *out)
{
    printf("START\n");
    print_long_list(in->shape, in->ndim);
    printf(" unbroadcast into ");
    print_long_list(shape, ndim);
    if (TensorBase_create_empty_like(in, out) < 0)
    {
        return -1;
    }
    if (!TensorBase_is_singleton(in))
    {
        memcpy(out->data, in->data, sizeof(scalar) * out->numel);
    }

    if (!TensorBase_same_shape(in->shape, shape))
    {
        int status;
        long dimension_diff = labs(in->ndim - ndim);
        printf("first one");

        if (dimension_diff != 0)
        {
            IndexArray summation_dims;
            long i = 0;

            for (; i < dimension_diff; i++)
            {
                summation_dims[i] = i;
            }
            for (; i < MAX_RANK; i++)
            {
                summation_dims[i] = -1;
            }
            status = TensorBase_aggregate(in, summation_dims, 0, out, SCALAR_AGG_SUM);
            if (status < 0)
            {
                printf("errororororor87686 %d", status);
                return status;
            }
        }

        IndexArray originally_ones;
        long d = 0;
        for (long i = 0; i < ndim; i++)
        {
            if (shape[i] == 1)
            {
                originally_ones[d] = i;
                d++;
                printf("axis: %ld, ", d);
            }
        }
        if (d > 0)
        {
            printf("d!=0");
            for (; d < MAX_RANK; d++)
            {
                originally_ones[d] = -1;
            }

            TensorBase temp;
            status = TensorBase_aggregate(out, originally_ones, 1, &temp, SCALAR_AGG_SUM);
            if (status < 0)
            {
                printf("errororororor %d", status);
                return status;
            }
            // Out has a pointer to data that is stale...we dont need it. The new pointer is the summed pointer in temp.
            // dealloc out and copy the data into out from temp.
            TensorBase_dealloc(out);
            memcpy(out, &temp, sizeof(TensorBase));
        }
    }
    else
    {
        printf("shape no different");
    }
    printf("END\n\n");
    return 0;
}