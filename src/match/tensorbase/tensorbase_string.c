#include "tensorbase.h"
#include "tensorbase_util.c"

static void TensorBase_to_string_data(TensorBase *tb, long curr_dim, long data_index, long *spaces, int print)
{
    // if previous char wasa  newline, print spaces according to # of [ - # of ] in previous line.
    if (print)
    {

        for (long i = 0; i < *spaces; i++)
        {
            printf(" ");
        }
    }

    if (curr_dim >= tb->ndim - 1)
    {
        printf("[");
        long i = 0;
        for (; i < tb->shape[curr_dim] - 1; i++)
        {
            printf("%.2f,", tb->data[data_index + i]);
        }
        printf("%.2f", tb->data[data_index + i]);
        printf("]");
        return;
    }

    printf("[");
    *spaces += 1;
    int should_print = 0;
    long i = 0;
    for (; i < tb->shape[curr_dim] - 1; i++)
    {
        TensorBase_to_string_data(tb, curr_dim + 1, data_index + tb->strides[curr_dim] * i, spaces, should_print);
        printf(",\n");
        should_print = 1;
    }
    TensorBase_to_string_data(tb, curr_dim + 1, data_index + tb->strides[curr_dim] * i, spaces, should_print);
    printf("]");
    *spaces -= 1;
}

static void TensorBase_to_string_attributes(TensorBase *tb)
{
    printf("ndim: %ld, numel: %ld, ", tb->ndim, tb->numel);
    printf("shape: (");
    for (long i = 0; i < tb->ndim; i++)
    {
        printf("%ld,", tb->shape[i]);
    }
    printf("), ");
    printf("strides: (");
    for (long i = 0; i < tb->ndim; i++)
    {
        printf("%ld,", tb->strides[i]);
    }
    printf(")\n");
}

void TensorBase_to_string(TensorBase *tb)
{
    if (TensorBase_is_singleton(tb))
    {
        scalar value;
        memcpy(&value, &(tb->data), sizeof(scalar));
        printf("Tensor(%.2f) ", value);
    }
    else
    {
        printf("Tensor(");
        long spaces = 7;
        TensorBase_to_string_data(tb, 0, 0, &spaces, 0);
        printf(")\n");
    }
    TensorBase_to_string_attributes(tb);
}