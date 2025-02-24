#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include "tensorbase.h"
#include "tensorbase_util.c"

// Returns an array of all of the indices in a TensorBase object to pull data from based on the subscripts.

StatusCode TensorBase_get(TensorBase *in, SubscriptArray subscripts, long num_subscripts, TensorBase *out)
{
    // Calculate the resulting shape of the output array and initialize new data array with malloc
    long new_numel;
    scalar *data;

    IndexArray curr_index;
    // Initialize curr_index to the starting position of all subscripts.
    for (long i = 0; i < num_subscripts; i++)
    {
        curr_index[i] = subscripts[i].start;
    }
    // Start increasing here and go down
    long curr_dim = num_subscripts-1;

    for (long out_data_index = 0; out_data_index < new_numel; out_data_index++)
    {
        // Convert index into data indices
        long in_data_index = TensorBase_convert_indices_to_data_index(curr_index);
        data[out_data_index] = in->data[in_data_index];
        
        // Update the curr_index Index array.
        curr_index[curr_dim]


    }

    return OK;
}

StatusCode TensorBase_set_scalar(TensorBase *in, SubscriptArray subscripts, long num_subscripts, scalar s)
{
    return NOT_IMPLEMENTED;
}

StatusCode TensorBase_set_tensorbase(TensorBase *in, SubscriptArray subscripts, long num_subscripts, TensorBase *t)
{
    return NOT_IMPLEMENTED;
}