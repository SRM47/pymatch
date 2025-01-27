#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include "tensorbase.h"

/*********************************************************
 *                      UTILITIES                        *
 *********************************************************/

static inline long max_long(long a, long b)
{
    return a > b ? a : b;
}

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

/*********************************************************
 *                    Alloc & Dealloc                    *
 *********************************************************/

int TensorBase_init(TensorBase *td, ShapeArray shape, long ndim)
{
    if (ndim > MAX_RANK || ndim < 0)
    {
        return -1;
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
            return -3;
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
    td->data = (scalar *)malloc(td->numel * sizeof(scalar));
    if (td->data == NULL)
    {
        // Memory error.
        return -2;
    }

    printf("\n~~EXPECTED Shape~~\n");
    print_long_list(shape, MAX_RANK);
    printf("\n~~EXPECTED NDIM~~\n");
    printf("%d", ndim);
    printf("\n~~NDIM~~\n");
    printf("%d", td->ndim);
    printf("\n~~Strides~~\n");
    print_long_list(td->strides, MAX_RANK);
    printf("\n~~Shape~~\n");
    print_long_list(td->shape, MAX_RANK);
    printf("\n~~NUMEL~~\n");
    printf("%d\n", td->numel);

    return 0;
}

void TensorBase_dealloc(TensorBase *td)
{
    if (td == NULL)
    {
        return;
    }

    if (td->data != NULL)
    {
        free(td->data);
        td->data = NULL;
    }

    td->numel = 0;
    td->ndim = 0;
    // Use memset to zero out shape and strides arrays safely.
    memset(td->shape, 0, MAX_RANK * sizeof(long));
    memset(td->strides, 0, MAX_RANK * sizeof(long));
}

/*********************************************************
 *                        Utility                        *
 *********************************************************/

static inline int TensorBase_is_singleton(TensorBase *t)
{
    return t->ndim == 0 && t->numel == 1;
}

static inline int TensorBase_compare_shape(ShapeArray a_shape, ShapeArray b_shape)
{
    return memcmp(a_shape, b_shape, MAX_RANK * sizeof(long)) == 0;
}

static int TensorBase_create_empty_like(TensorBase *in, TensorBase *out)
{
    if (in == NULL || out == NULL)
    {
        return -1; // Invalid input or output tensor
    }

    if (out->data != NULL)
    {
        free(out->data);
        out->data = NULL;
    }
    out->data = (scalar *)malloc(in->numel * sizeof(scalar));
    if (out->data == NULL)
    {
        return -1;
    }
    out->numel = in->numel;
    out->ndim = in->ndim;
    memcpy(out->shape, in->shape, MAX_RANK * sizeof(long));
    memcpy(out->strides, in->strides, MAX_RANK * sizeof(long));
    return 0;
}

/*********************************************************
 *                     String Methods                    *
 *********************************************************/

static void TensorBase_to_string_helper(TensorBase *tb, long curr_dim, long data_index, long *spaces, int print)
{
    // if previous char wasa  newline, print spaces according to # of [ - # of ] in previous line.
    if (print)
    {
        long num_spaces = *spaces;
        printf("%d", num_spaces);
        for (long i = 0; i < num_spaces; i++)
        {
            printf("[]");
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
        TensorBase_to_string_helper(tb, curr_dim + 1, data_index + tb->strides[curr_dim] * i, spaces, should_print);
        printf(",\n");
        should_print = 1;
    }
    TensorBase_to_string_helper(tb, curr_dim + 1, data_index + tb->strides[curr_dim] * i, spaces, should_print);
    printf("]");
    *spaces -= 1;
}

void TensorBase_to_string(TensorBase *td)
{
    int spaces = 0;
    TensorBase_to_string_helper(td, 0, 0, &spaces, 0);
}

// void TensorBase_to_string(TensorBase *td, char *buffer, size_t buffer_size)
// {
//     int bytes_written = snprintf(buffer, buffer_size, "[");
//     buffer_size -= bytes_written;
//     buffer += bytes_written;

//     for (size_t index = 0; index < td->numel && buffer_size > 0; index++)
//     {
//         const char *sep = index < td->numel - 1 ? ", " : "";
//         bytes_written = snprintf(buffer, buffer_size, "%f%s", td->data[index], sep);
//         buffer_size -= bytes_written;
//         buffer += bytes_written;
//     }

//     // TODO: Check for buffer overflow.
//     snprintf(buffer, buffer_size, "]");
// }

/*********************************************************
 *                     Braodcasting                      *
 *********************************************************/

static int TensorBase_get_broadcast_shape(TensorBase *a, TensorBase *b, ShapeArray broadcast_shape, int *broadcast_ndim)
{
    // Initialize broadcast_shape with -1 to indicate dimensions that haven't been determined yet.
    for (size_t i = 0; i < MAX_RANK; i++)
    {
        broadcast_shape[i] = -1;
    }

    // Determine the maximum rank (number of dimensions)
    *broadcast_ndim = max_long(a->ndim, b->ndim);

    // Broadcast dimensions
    long a_index = a->ndim - 1;
    long b_index = b->ndim - 1;
    long out_index = *broadcast_ndim - 1;

    for (; out_index >= 0; out_index--, a_index--, b_index--)
    {
        if ((a_index >= 0 && a->shape[a_index] == 1) || (a_index < 0 && b_index >= 0))
        {
            broadcast_shape[out_index] = b->shape[b_index];
        }
        else if ((b_index >= 0 && b->shape[b_index] == 1) || (b_index < 0 && a_index >= 0))
        {
            broadcast_shape[out_index] = a->shape[a_index];
        }
        else if (a->shape[a_index] == b->shape[b_index])
        {
            broadcast_shape[out_index] = a->shape[a_index];
        }
        else
        {
            // Incompatible shapes.
            return -1;
        }
    }

    return 0;
}

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

/*********************************************************
 *                    Linear Algebra                     *
 *********************************************************/

void print_double_list(const double *list, size_t size)
{
    // Check if the list is NULL
    if (list == NULL)
    {
        printf("List is NULL.\n");
        return;
    }

    printf("Size of list: %zu\n", size);
    printf("List of doubles: [");
    for (size_t i = 0; i < size; i++)
    {
        printf("%f", list[i]);
        if (i < size - 1)
        {
            printf(", ");
        }
    }
    printf("]\n");
}

void print_long_list(const long *list, size_t size)
{
    // Check if the list is NULL
    if (list == NULL)
    {
        printf("List is NULL.\n");
        return;
    }

    printf("Size of list: %zu\n", size);
    printf("List of doubles: [");
    for (size_t i = 0; i < size; i++)
    {
        printf("%d", list[i]);
        if (i < size - 1)
        {
            printf(", ");
        }
    }
    printf("]\n");
}

int TensorBase_binary_op_tensorbase_tensorbase(TensorBase *a, TensorBase *b, TensorBase *out, scalar (*op)(scalar, scalar))
{
    // Check if a is a singleton.
    if (TensorBase_is_singleton(a))
    {
        scalar s = *(a->data);
        return TensorBase_binary_op_scalar_tensorbase(b, s, out, op);
    }

    // Check if b is a singleton.
    if (TensorBase_is_singleton(b))
    {
        scalar s = *(b->data);
        return TensorBase_binary_op_tensorbase_scalar(a, s, out, op);
    }

    // Check if the two tensorbase structures don't have the same dimensions
    if (TensorBase_compare_shape(a->shape, b->shape))
    {
        // They have the same shape.
        if (TensorBase_create_empty_like(a, out) == -1)
        {
            return -1;
        }
        for (long i = 0; i < out->numel; i++)
        {
            out->data[i] = op(a->data[i], b->data[i]);
        }
    }
    else
    {
        // They don't have the same shape must *attempt to* broadcast.
        ShapeArray broadcasted_tensor_shape;
        long broadcasted_tensor_ndim;
        if (TensorBase_get_broadcast_shape(a, b, broadcasted_tensor_shape, &broadcasted_tensor_ndim) == -1)
        {
            // Incompatible broadcasting shapes.
            return -1;
        }

        if (TensorBase_init(out, broadcasted_tensor_shape, broadcasted_tensor_ndim) == -1)
        {
            return -1;
        }
        printf("\na: %p to %p\nb: %p to %p\nout: %p to %p\n", a->data, a->data + (a->numel * sizeof(scalar)), b->data, b->data + (b->numel * sizeof(scalar)), out->data, out->data + (out->numel * sizeof(scalar)));
        printf("before broadcast operation\n");
        print_double_list(out->data, out->numel);
        // Loop through each element in the broadcasted tensor's data.
        for (long broadcasted_data_index = 0; broadcasted_data_index < out->numel; broadcasted_data_index++)
        {
            // For each element in the data index, calculate the corresponding
            // element in each of the input tensors.
            long a_data_index = 0;
            long b_data_index = 0;

            long a_dim = a->ndim - 1;
            long b_dim = b->ndim - 1;
            long broadcast_dim = out->ndim - 1;

            long temp = broadcasted_data_index;

            for (; broadcast_dim >= 0; broadcast_dim--, a_dim--, b_dim--)
            {
                // (2,5,6)
                long broadcast_coordinate_at_curr_dim = temp % out->shape[broadcast_dim];
                temp /= out->shape[broadcast_dim];

                if (a_dim >= 0 && a->shape[a_dim] > 1)
                {
                    a_data_index += broadcast_coordinate_at_curr_dim * a->strides[a_dim];
                }

                if (b_dim >= 0 && b->shape[b_dim] > 1)
                {
                    b_data_index += broadcast_coordinate_at_curr_dim * b->strides[b_dim];
                }
            }
            printf("a->data[%d] = %f\n", a_data_index, a->data[a_data_index]);
            printf("b->data[%d] = %f\n", b_data_index, b->data[b_data_index]);

            out->data[broadcasted_data_index] = op(a->data[a_data_index], b->data[b_data_index]);

            printf("out->data[%d]: %f\n", broadcasted_data_index, out->data[broadcasted_data_index]);
        }
        printf("after broadcast operation\n");
        print_double_list(out->data, out->numel);
    }

    return 0;
}

int TensorBase_binary_op_tensorbase_scalar(TensorBase *a, scalar s, TensorBase *out, scalar (*op)(scalar, scalar))
{
    if (TensorBase_create_empty_like(a, out) == -1)
    {
        return -1;
    }
    for (long i = 0; i < out->numel; i++)
    {
        out->data[i] = op(a->data[i], s);
    }
    return 0;
}

int TensorBase_binary_op_scalar_tensorbase(TensorBase *a, scalar s, TensorBase *out, scalar (*op)(scalar, scalar))
{
    if (TensorBase_create_empty_like(a, out) == -1)
    {
        return -1;
    }
    for (long i = 0; i < out->numel; i++)
    {
        out->data[i] = op(s, a->data[i]);
    }
    return 0;
}

int TensorBase_unary_op_inplace(TensorBase *in, scalar (*op)(scalar))
{
    if (in->data == NULL)
    {
        return -1;
    }
    for (long i = 0; i < in->numel; i++)
    {
        in->data[i] = op(in->data[i]);
    }
    return 0;
}

int TensorBase_unary_op(TensorBase *in, TensorBase *out, scalar (*op)(scalar))
{
    if (TensorBase_create_empty_like(in, out) == -1)
    {
        return -1;
    }
    for (long i = 0; i < out->numel; i++)
    {
        out->data[i] = op(in->data[i]);
    }
    return 0;
}

// TODO: Implement
int TensorBase_get_matrix_multiplication_shape(TensorBase *a, TensorBase *b, ShapeArray *out)
{
    return -2;
    // TODO: relax the requirement for 2D tensors
    if (a->ndim != 2 || b->ndim != 2)
    {
        return -1;
    }

    if (a->shape[1] != b->shape[0])
    {
        return -1;
    }

    (*out)[0] = a->shape[0];
    (*out)[1] = b->shape[1];

    return 1;
}

// TODO: Implement
int TensorBase_matrix_multiply(TensorBase *a, TensorBase *b, TensorBase *out)
{
    return -2;

    for (long i = 0; i < a->shape[0]; i++)
    {
        for (long j = 0; j < b->shape[1]; j++)
        {
            scalar sum = 0;
            for (long k = 0; k < a->shape[1]; k++)
            {
                sum += a->data[i * a->shape[1] + k] * b->data[k * b->shape[1] + j];
            }
            out->data[i * out->shape[1] + j] = sum;
        }
    }
}

scalar scalar_add(scalar a, scalar b) { return a + b; }
scalar scalar_sub(scalar a, scalar b) { return a - b; }
scalar scalar_mult(scalar a, scalar b) { return a * b; }
scalar scalar_floordiv(scalar a, scalar b) { return floor(a / b); }
scalar scalar_truediv(scalar a, scalar b) { return a / b; }
scalar scalar_power(scalar a, scalar b) { return pow(a, b); }
scalar scalar_negative(scalar a) { return -a; }
scalar scalar_absolute(scalar a) { return fabs(a); }

/*********************************************************
 *                      Aggregation                      *
 *********************************************************/

// TODO: Implement
int TensorBase_aggregate(TensorBase *in, IndexArray dim, int keepdim, TensorBase *out, scalar (*aggregate)(scalar *, long)) { return -2; }

/*********************************************************
 *                     Manipulation                      *
 *********************************************************/

int TensorBase_permute_inplace(TensorBase *in, IndexArray permutation) { return -2; }
int TensorBase_permute(TensorBase *in, IndexArray permutation, TensorBase *out) { return -2; }

int TensorBase_reshape_inplace(TensorBase *in, ShapeArray shape) { return -2; }
int TensorBase_reshape(TensorBase *in, ShapeArray shape, TensorBase *out) { return -2; }

int TensorBase_fill_(TensorBase *in, scalar fill_value)
{
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
int TensorBase_randn_(TensorBase *in) { return -2; }

int TensorBase_item(TensorBase *t, scalar *item)
{
    if (t->numel != 1)
    {
        return -1;
    }
    *item = *(t->data);
    return 0;
}
