#pragma once

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include "tensorbase.h"

// C macro do{}while(0).
#define RETURN_IF_ERROR(x) ({ StatusCode _status = x; if (_status != OK) { return _status; } })

static inline long max_long(long a, long b)
{
    return a > b ? a : b;
}

static inline int TensorBase_is_singleton(TensorBase *t)
{
    return t->ndim == 0;
}

static inline int TensorBase_same_shape(ShapeArray a_shape, ShapeArray b_shape)
{
    return memcmp(a_shape, b_shape, MAX_RANK * sizeof(long)) == 0;
}

static StatusCode TensorBase_create_empty_like(TensorBase *in, TensorBase *out)
{
    // Assumes out->data doesn't point to any alocated memory.
    if (in == NULL || out == NULL)
    {
        return NULL_INPUT_ERR; // Invalid input or output tensor
    }

    memcpy(out, in, sizeof(TensorBase));

    if (!TensorBase_is_singleton(in))
    {
        out->data = (scalar *)malloc(in->numel * sizeof(scalar));
        if (out->data == NULL)
        {
            return MALLOC_ERR;
        }
    }

    return OK;
}

static void print_long_list(const long *list, size_t size)
{
    printf("[");
    for (long i = 0; i < size; i++)
    {
        printf("%ld, ", list[i]);
    }
    printf("]\n");
}

static StatusCode TensorBase_get_broadcast_shape(ShapeArray a_shape, long a_ndim, ShapeArray b_shape, long b_ndim, ShapeArray broadcast_shape, long *broadcast_ndim)
{
    // Initialize broadcast_shape with -1 to indicate dimensions that haven't been determined yet.
    for (size_t i = 0; i < MAX_RANK; i++)
    {
        broadcast_shape[i] = -1;
    }

    // Determine the maximum rank (number of dimensions)
    *broadcast_ndim = max_long(a_ndim, b_ndim);

    // Broadcast dimensions
    long a_index = a_ndim - 1;
    long b_index = b_ndim - 1;
    long out_index = *broadcast_ndim - 1;

    for (; out_index >= 0; out_index--, a_index--, b_index--)
    {
        if ((a_index >= 0 && a_shape[a_index] == 1) || (a_index < 0 && b_index >= 0))
        {
            broadcast_shape[out_index] = b_shape[b_index];
        }
        else if ((b_index >= 0 && b_shape[b_index] == 1) || (b_index < 0 && a_index >= 0))
        {
            broadcast_shape[out_index] = a_shape[a_index];
        }
        else if (a_shape[a_index] == b_shape[b_index])
        {
            broadcast_shape[out_index] = a_shape[a_index];
        }
        else
        {
            return INCOMPATABLE_BROASCAST_SHAPES;
        }
    }

    return OK;
}
