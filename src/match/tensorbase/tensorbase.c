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

static void index_to_indices(long index, ShapeArray shape, long ndim, IndexArray out)
{
    for (long i = ndim - 1; i >= 0; i--)
    {
        out[i] = index % shape[i];
        index /= shape[i];
    }
}

static long indices_to_index(IndexArray indices, StrideArray strides, long ndim)
{
    long index = 0;
    for (long i = 0; i < ndim; i++)
    {
        index += indices[i] * strides[i];
    }
    return index;
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
    memset(td->shape, -1, sizeof(ShapeArray));
    memset(td->strides, 0, sizeof(StrideArray));
    // Allocate the memory, but do not initialize it's values.
    td->data = (scalar *)malloc(td->numel * sizeof(scalar));
    if (td->data == NULL)
    {
        // Memory error.
        return -1;
    }

    // Calculate Tensorbase shape and number of elements.
    long numel_for_stride = 1;
    for (int i = 0; i < ndim; i++)
    {
        long dim = shape[i];
        if (dim < 0)
        {
            // All dimensions must be >= 0 (-1 indicates no-dimension).
            return -1;
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

    return 0;
}

void TensorBase_dealloc(TensorBase *td)
{
    if (td->data != NULL)
    {
        free(td->data);
    }
    td->data = NULL;
    td->numel = 0;
    td->ndim = 0;
    memset(td->shape, 0, sizeof(ShapeArray));
    memset(td->strides, 0, sizeof(StrideArray));
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
    return memcmp(a_shape, b_shape, MAX_RANK);
}

static int TensorBase_create_empty_like(TensorBase *in, TensorBase *out)
{
    if (out->data != NULL)
    {
        free(out->data);
    }
    out->data = (scalar *)malloc(in->numel * sizeof(scalar));
    if (out->data == NULL)
    {
        return -1;
    }
    out->numel = in->numel;
    out->ndim = in->ndim;
    memcpy(out->shape, in->shape, sizeof(in->shape));
    memcpy(out->strides, in->strides, sizeof(in->strides));
    return 0;
}

/*********************************************************
 *                     String Methods                    *
 *********************************************************/
// TODO: Turn into stringify
void TensorBase_to_string(TensorBase *td, char *buffer, size_t buffer_size)
{
    int bytes_written = snprintf(buffer, buffer_size, "[");
    buffer_size -= bytes_written;
    buffer += bytes_written;

    for (size_t index = 0; index < td->numel && buffer_size > 0; index++)
    {
        const char *sep = index < td->numel - 1 ? ", " : "";
        bytes_written = snprintf(buffer, buffer_size, "%f%s", td->data[index], sep);
        buffer_size -= bytes_written;
        buffer += bytes_written;
    }

    // TODO: Check for buffer overflow.
    snprintf(buffer, buffer_size, "]");
}

/*********************************************************
 *                     Braodcasting                      *
 *********************************************************/

static int TensorBase_get_broadcast_shape(TensorBase *a, TensorBase *b, ShapeArray broadcast_shape, int *broadcast_ndim)
{
    // Initialize broadcast_shape with -1 to indicate dimensions that haven't been determined yet.
    memset(broadcast_shape, -1, MAX_RANK * sizeof(long));

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
        scalar s = *b->data;
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
            out->data[i] = op(a->data[i], a->data[i]);
        }
    }
    else
    {
        // They don't have the same shape must *attempt to* broadcast.

        ShapeArray broadcast_shape;
        int broadcast_ndim;
        if (TensorBase_get_broadcast_shape(a, b, broadcast_shape, &broadcast_ndim) == -1)
        {
            // Incompatible broadcasting shapes.
            return -1;
        }

        if (TensorBase_init(out, broadcast_shape, broadcast_ndim) == -1)
        {
            return -1;
        }

        for (long broadcast_index = 0; broadcast_index < out->numel; broadcast_index++)
        {
            // Translate from broadcast data index to a and b data index.
            long a_index = 0;
            long b_index = 0;

            long a_broadcast_dim_offset = out->ndim - a->ndim;
            long b_broadcast_dim_offset = out->ndim - b->ndim;

            for (int broadcast_dim_index = out->ndim - 1; broadcast_dim_index >= 0; broadcast_dim_index--)
            {
                long a_dim = broadcast_dim_index - a_broadcast_dim_offset;
                long b_dim = broadcast_dim_index - b_broadcast_dim_offset;

                long broadcast_shape_at_curr_dim = broadcast_index % out->shape[broadcast_dim_index];
                broadcast_index /= out->shape[broadcast_dim_index];

                if (a_dim >= 0 && a->shape[a_dim] > 1)
                {
                    a_index += broadcast_shape_at_curr_dim * a->strides[a_dim];
                }

                if (b_dim >= 0 && b->shape[b_dim] > 1)
                {
                    b_index += broadcast_shape_at_curr_dim * b->strides[b_dim];
                }
            }

            out->data[broadcast_index] = op(a->data[a_index], b->data[b_index]);
        }
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

int TensorBase_fill_(TensorBase *in, scalar fill_value) { return -2; }
int TensorBase_randn_(TensorBase *in) { return -2; }
