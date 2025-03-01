#include "tensorbase.h"
#include "tensorbase_util.c"

static void pretty_print_tensor_data_array(TensorBase *tb, long curr_dim, long data_index, long *offset, bool was_previous_char_newline)
{
    if (was_previous_char_newline)
    {
        // Print `offset` number of spaces.
        printf("%*s", (int)*offset - 1, "");
    }

    long dimension_size = tb->shape[curr_dim];

    if (curr_dim >= tb->ndim - 1)
    {
        printf("[");
        for (long i = 0; i < dimension_size; i++)
        {
            printf("%.2f", tb->data[data_index + i]);
            if (i < dimension_size - 1)
            {
                printf(",");
            }
        }
        printf("]");
        return;
    }

    printf("[");
    *offset += 1;
    for (long i = 0; i < dimension_size; i++)
    {
        long next_data_index = data_index + tb->strides[curr_dim] * i;
        was_previous_char_newline = (i != 0); // The first iteration of the loop did not have a preceeding newline.
        pretty_print_tensor_data_array(tb, curr_dim + 1, next_data_index, offset, was_previous_char_newline);
        if (i < dimension_size - 1)
        {
            printf(",\n");
        }
    }
    printf("]");
    *offset -= 1;
}

static void pretty_print_tensor_attributes(TensorBase *tb)
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
    if (tb == NULL)
    {
        return;
    }
    // Print the tensor contents.
    if (TensorBase_is_singleton(tb))
    {
        scalar value;
        memcpy(&value, &(tb->data), sizeof(scalar));
        printf("Tensor(%.2f) ", value);
    }
    else
    {
        const char *tensor_name = "tensor";
        long initial_data_offset = sizeof(tensor_name) + 1;

        printf(tensor_name);
        printf("(");
        pretty_print_tensor_data_array(tb, 0, 0, &initial_data_offset, false);
        printf(")\n");
    }
    // Print the shape, strides, ndim, and numel.
    pretty_print_tensor_attributes(tb);
}