#include "tensorbase.h"

// ----------------------------------------------------------------
// ▗▖ ▗▖       █  ▗▄▖    █         █
// ▐▌ ▐▌ ▐▌    ▀  ▝▜▌    ▀   ▐▌    ▀
// ▐▌ ▐▌▐███  ██   ▐▌   ██  ▐███  ██   ▟█▙ ▗▟██▖
// ▐▌ ▐▌ ▐▌    █   ▐▌    █   ▐▌    █  ▐▙▄▟▌▐▙▄▖▘
// ▐▌ ▐▌ ▐▌    █   ▐▌    █   ▐▌    █  ▐▛▀▀▘ ▀▀█▖
// ▝█▄█▘ ▐▙▄ ▗▄█▄▖ ▐▙▄ ▗▄█▄▖ ▐▙▄ ▗▄█▄▖▝█▄▄▌▐▄▄▟▌
//  ▝▀▘   ▀▀ ▝▀▀▀▘  ▀▀ ▝▀▀▀▘  ▀▀ ▝▀▀▀▘ ▝▀▀  ▀▀▀
// ----------------------------------------------------------------

static inline long max_long(long a, long b)
{
    return a > b ? a : b;
}

// Convert a linear index to a multi-dimensional index
// NOTE: this function does not check for out-of-bounds indices
static void index_to_indices(long index, ShapeArray shape, long ndim, IndexArray out)
{
    for (long i = ndim - 1; i >= 0; i--)
    {
        out[i] = index % shape[i];
        index /= shape[i];
    }
}

// Convert a multi-dimensional index to a linear index
// NOTE: this function does not check for out-of-bounds indices
static long indices_to_index(IndexArray indices, StrideArray strides, long ndim)
{
    long index = 0;
    for (long i = 0; i < ndim; i++)
    {
        index += indices[i] * strides[i];
    }
    return index;
}

// TODO: https://www.pcg-random.org/download.html

typedef struct
{
    scalar a;
    scalar b;
} randn_pair;

// Box-Muller method for generating normally distributed random numbers
// https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform#C++
randn_pair randn(scalar mu, scalar sigma)
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

// ----------------------------------------------------------------
// ▗▄▄▄▖                              ▗▄▄▖
// ▝▀█▀▘                              ▐▛▀▜▌
//   █   ▟█▙ ▐▙██▖▗▟██▖ ▟█▙  █▟█▌     ▐▌ ▐▌ ▟██▖▗▟██▖ ▟█▙
//   █  ▐▙▄▟▌▐▛ ▐▌▐▙▄▖▘▐▛ ▜▌ █▘       ▐███  ▘▄▟▌▐▙▄▖▘▐▙▄▟▌
//   █  ▐▛▀▀▘▐▌ ▐▌ ▀▀█▖▐▌ ▐▌ █        ▐▌ ▐▌▗█▀▜▌ ▀▀█▖▐▛▀▀▘
//   █  ▝█▄▄▌▐▌ ▐▌▐▄▄▟▌▝█▄█▘ █        ▐▙▄▟▌▐▙▄█▌▐▄▄▟▌▝█▄▄▌
//   ▀   ▝▀▀ ▝▘ ▝▘ ▀▀▀  ▝▀▘  ▀        ▝▀▀▀  ▀▀▝▘ ▀▀▀  ▝▀▀
// ----------------------------------------------------------------

static int TensorBase_init(TensorBase *td, ShapeArray shape, long ndim)
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

static void TensorBase_dealloc(TensorBase *td)
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

// ----------------------------------------------------------------
// ▗▄▄▖                   ▗▖                      █
// ▐▛▀▜▌                  ▐▌                ▐▌    ▀
// ▐▌ ▐▌ █▟█▌ ▟█▙  ▟██▖ ▟█▟▌ ▟██▖ ▟██▖▗▟██▖▐███  ██  ▐▙██▖ ▟█▟▌
// ▐███  █▘  ▐▛ ▜▌ ▘▄▟▌▐▛ ▜▌▐▛  ▘ ▘▄▟▌▐▙▄▖▘ ▐▌    █  ▐▛ ▐▌▐▛ ▜▌
// ▐▌ ▐▌ █   ▐▌ ▐▌▗█▀▜▌▐▌ ▐▌▐▌   ▗█▀▜▌ ▀▀█▖ ▐▌    █  ▐▌ ▐▌▐▌ ▐▌
// ▐▙▄▟▌ █   ▝█▄█▘▐▙▄█▌▝█▄█▌▝█▄▄▌▐▙▄█▌▐▄▄▟▌ ▐▙▄ ▗▄█▄▖▐▌ ▐▌▝█▄█▌
// ▝▀▀▀  ▀    ▝▀▘  ▀▀▝▘ ▝▀▝▘ ▝▀▀  ▀▀▝▘ ▀▀▀   ▀▀ ▝▀▀▀▘▝▘ ▝▘ ▞▀▐▌
//                                                         ▜█▛▘
// ----------------------------------------------------------------

// Fill-in the shape and strides of the (temporary) output tensors
static int TensorBase_broadcast_for_binop(TensorBase *a_in, TensorBase *b_in, TensorBase *a_out, TensorBase *b_out)
{
    a_out->numel = a_in->numel;
    b_out->numel = b_in->numel;

    // The new (temporary) objects share data with the input objects
    a_out->data = a_in->data;
    b_out->data = b_in->data;

    long a_index = a_in->ndim - 1;
    long b_index = b_in->ndim - 1;
    long out_index = MAX_RANK - 1;

    long max_dim = max_long(a_in->ndim, b_in->ndim);
    a_out->ndim = max_dim;
    b_out->ndim = max_dim;

    // Fill final slots of out shape with zeros
    for (; out_index > max_dim - 1; out_index--)
    {
        a_out->shape[out_index] = 0;
        b_out->shape[out_index] = 0;
    }

    // Broadcast remaining shape dimensions
    for (; out_index >= 0; out_index--, a_index--, b_index--)
    {
        if ((a_index >= 0 && a_in->shape[a_index] == 1) || (a_index < 0 && b_index >= 0))
        {
            a_out->shape[out_index] = b_in->shape[b_index];
            b_out->shape[out_index] = b_in->shape[b_index];
        }
        else if ((b_index >= 0 && b_in->shape[b_index] == 1) || (b_index < 0 && a_index >= 0))
        {
            a_out->shape[out_index] = a_in->shape[a_index];
            b_out->shape[out_index] = a_in->shape[a_index];
        }
        else if (a_in->shape[a_index] == b_in->shape[b_index])
        {
            a_out->shape[out_index] = a_in->shape[a_index];
            b_out->shape[out_index] = a_in->shape[a_index];
        }
        else
        {
            // Incompatible shapes
            return -1;
        }
    }

    // Set stride of (temporary) tensors (use 0 for dimensions of size 1)
    // TODO: Handle when shape[i] == 0.
    long a_stride = a_out->numel;
    long b_stride = b_out->numel;

    for (out_index = 0; out_index < max_dim; out_index++)
    {
        long a_dim = a_out->shape[out_index];
        a_stride /= a_dim;
        a_out->strides[out_index] = a_dim > 0 ? a_stride : 0;

        long b_dim = b_out->shape[out_index];
        b_stride /= b_dim;
        b_out->strides[out_index] = b_dim > 0 ? b_stride : 0;
    }

    // Fill in remaining strides with zero
    for (; out_index < MAX_RANK; out_index++)
    {
        a_out->strides[out_index] = 0;
        b_out->strides[out_index] = 0;
    }

    // printf("\n\na_ndim: %ld, b_ndim: %ld\n", a_out->ndim, b_out->ndim);
    // printf("a_out->shape: [%ld, %ld], b_out->shape: [%ld, %ld]\n", a_out->shape[0], a_out->shape[1], b_out->shape[0], b_out->shape[1]);
    // printf("a_out->numel: %ld, b_out->numel: %ld\n", a_out->numel, b_out->numel);
    // printf("a_out->strides: [%ld, %ld], b_out->strides: [%ld, %ld]\n", a_out->strides[0], a_out->strides[1], b_out->strides[0], b_out->strides[1]);

    return 0;
}

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
            // Incompatible shapes
            return -1;
        }
    }

    return 0;
}

// ----------------------------------------------------------------
// ▗▖     █         ▄  ▗▄▖
// ▐▌     ▀        ▐█▌ ▝▜▌
// ▐▌    ██  ▐▙██▖ ▐█▌  ▐▌   ▟█▟▌
// ▐▌     █  ▐▛ ▐▌ █ █  ▐▌  ▐▛ ▜▌
// ▐▌     █  ▐▌ ▐▌ ███  ▐▌  ▐▌ ▐▌
// ▐▙▄▄▖▗▄█▄▖▐▌ ▐▌▗█ █▖ ▐▙▄ ▝█▄█▌
// ▝▀▀▀▘▝▀▀▀▘▝▘ ▝▘▝▘ ▝▘  ▀▀  ▞▀▐▌
//                           ▜█▛▘
// ----------------------------------------------------------------
static inline int TensorBase_is_singleton(TensorBase *t)
{
    return t->ndim == 0 && t->numel == 1;
}

static int TensorBase_copy_only_allocate(TensorBase *a, TensorBase *out)
{
    if (out->data == NULL)
    {
        out->data = (scalar *)malloc(a->numel * sizeof(scalar));
        if (out->data == NULL)
        {
            return -1;
        }
    }
    out->numel = a->numel;
    out->ndim = a->ndim;
    memcpy(out->shape, a->shape, sizeof(a->shape));
    memcpy(out->strides, a->strides, sizeof(a->strides));
    return 0;
}

static inline int TensorBase_same_shape(ShapeArray a_shape, ShapeArray b_shape)
{
    return memcmp(a_shape, b_shape, MAX_RANK);
}

/*

self_coordinates = [0] * len(
            self.shape
        )  # Initialize a list to store the translated coordinates.

        # Iterate over the dimensions of the existing tensor in reverse order.
        for existing_dimension_index in range(len(self.shape) - 1, -1, -1):
            # If the shape at the current index of the existing tensor is 1, set the coordinate to 0 in that dimension.
            # Otherwise, use the corresponding coordinate from the new tensor coordinates.
            if self.shape[existing_dimension_index] == 1:
                self_coordinates[existing_dimension_index] = 0
            else:
                # Calculate the index in the new tensor coordinates corresponding to the current dimension of the existing tensor.
                new_tensor_dimension_index = (
                    existing_dimension_index
                    + len(new_tensor_coordinates)
                    - len(self.shape)
                )
                self_coordinates[existing_dimension_index] = new_tensor_coordinates[
                    new_tensor_dimension_index
                ]

        return tuple(
            self_coordinates
        )  # Return the translated coordinates of the existing tensor.

        // Convert a linear index to a multi-dimensional index
// NOTE: this function does not check for out-of-bounds indices
static void index_to_indices(long index, ShapeArray shape, long ndim, IndexArray out)
{
    for (long i = ndim - 1; i >= 0; i--)
    {
        out[i] = index % shape[i];
        index /= shape[i];
    }
}

// Convert a multi-dimensional index to a linear index
// NOTE: this function does not check for out-of-bounds indices
static long indices_to_index(IndexArray indices, StrideArray strides, long ndim)
{
    long index = 0;
    for (long i = 0; i < ndim; i++)
    {
        index += indices[i] * strides[i];
    }
    return index;
}

*/

// a `op` b
static int TensorBase_binary_op_tensorbase_tensorbase(TensorBase *a, TensorBase *b, TensorBase *out, scalar (*op)(scalar, scalar))
{
    // Check if a is a singleton.
    if (TensorBase_is_singleton(a))
    {
        scalar s = *a->data;
        return TensorBase_binary_op_scalar_tensorbase(b, s, out, op);
    }

    // Check if b is a singleton.
    if (TensorBase_is_singleton(b))
    {
        scalar s = *b->data;
        return TensorBase_binary_op_tensorbase_scalar(a, s, out, op);
    }

    // Check if the two tensorbase structures don't have the same dimensions
    if (Tensorbase_same_shape(a->shape, b->shape))
    {
        // They have the same shape.
        if (TensorBase_copy_only_allocate(a, out) == -1)
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

// tensorbase `op` scalar
static int TensorBase_binary_op_tensorbase_scalar(TensorBase *a, scalar s, TensorBase *out, scalar (*op)(scalar, scalar))
{
    if (TensorBase_copy_only_allocate(a, out) == -1)
    {
        return -1;
    }
    for (long i = 0; i < out->numel; i++)
    {
        out->data[i] = op(a->data[i], s);
    }
    return 0;
}

// scalar `op` tensorbase
static int TensorBase_binary_op_scalar_tensorbase(TensorBase *a, scalar s, TensorBase *out, scalar (*op)(scalar, scalar))
{
    if (TensorBase_copy_only_allocate(a, out) == -1)
    {
        return -1;
    }
    for (long i = 0; i < out->numel; i++)
    {
        out->data[i] = op(s, a->data[i]);
    }
    return 0;
}

static int TensorBase_unary_op_inplace(TensorBase *a, scalar (*op)(scalar))
{
    if (a->data == NULL)
    {
        return -1;
    }
    for (long i = 0; i < a->numel; i++)
    {
        a->data[i] = op(a->data[i]);
    }
    return 0;
}

static int TensorBase_unary_op(TensorBase *a, TensorBase *out, scalar (*op)(scalar))
{
    if (TensorBase_copy_only_allocate(a, out) == -1)
    {
        return -1;
    }
    for (long i = 0; i < out->numel; i++)
    {
        out->data[i] = op(a->data[i]);
    }
    return 0;
}

// TODO: turn into TensorBase_binop_tensor_tensor?
static void TensorBase_add_tensor_tensor(TensorBase *a, TensorBase *b, TensorBase *out)
{
    IndexArray a_indices = {0};
    IndexArray b_indices = {0};

    // printf("a->ndim: %ld, b->ndim: %ld, out->ndim: %ld\n", a->ndim, b->ndim, out->ndim);
    // printf("a->shape: [%ld, %ld], b->shape: [%ld, %ld], out->shape: [%ld, %ld]\n", a->shape[0], a->shape[1], b->shape[0], b->shape[1], out->shape[0], out->shape[1]);
    // printf("a->numel: %ld, b->numel: %ld, out->numel: %ld\n", a->numel, b->numel, out->numel);
    // printf("a->strides: [%ld, %ld], b->strides: [%ld, %ld], out->strides: [%ld, %ld]\n", a->strides[0], a->strides[1], b->strides[0], b->strides[1], out->strides[0], out->strides[1]);

    for (long i = 0; i < out->numel; i++)
    {
        // TODO: need a utility to make this faster (no need to two functions)
        index_to_indices(i, a->shape, a->ndim, a_indices);
        long a_index = indices_to_index(a_indices, a->strides, a->ndim);

        index_to_indices(i, b->shape, b->ndim, b_indices);
        long b_index = indices_to_index(b_indices, b->strides, b->ndim);

        // printf("a_index: %ld, b_index: %ld\n", a_index, b_index);
        // printf("a_indices: [%ld, %ld], b_indices: [%ld, %ld]\n", a_indices[0], a_indices[1], b_indices[0], b_indices[1]);

        out->data[i] = a->data[a_index] + b->data[b_index];
    }
}

// TODO: turn into TensorBase_binop_tensor_scalar?
static void TensorBase_add_tensor_scalar(TensorBase *t, scalar s, TensorBase *out)
{
    for (long i = 0; i < out->numel; i++)
    {
        // TODO: check how much slower this is if the user passes in the
        // operator, and we put a switch statement in the loop (check godbolt?)
        out->data[i] = t->data[i] + s;
    }
}

static void TensorBase_div_scalar_tensor(scalar s, TensorBase *t, TensorBase *out)
{
    for (long i = 0; i < out->numel; i++)
    {
        out->data[i] = s / t->data[i];
    }
}

static void TensorBase_neg(TensorBase *t, TensorBase *out)
{
    for (long i = 0; i < out->numel; i++)
    {
        out->data[i] = -t->data[i];
    }
}

static void TensorBase_sigmoid(TensorBase *t, TensorBase *out)
{
    for (long i = 0; i < out->numel; i++)
    {
        out->data[i] = 1.0 / (1.0 + exp(-t->data[i]));
    }
}

static long TensorBase_get_matrix_multiplication_shape(TensorBase *a, TensorBase *b, ShapeArray *out)
{
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

// void mat_mul_jki(int n, float *A, float *B, float *C)
// {
//     for (int j = 0; j < n; j++)
//         for (int k = 0; k < n; k++)
//             for (int i = 0; i < n; i++)
//                 C[i + j * n] += A[i + k * n] * B[k + j * n];
// }

static void TensorBase_matrix_multiply(TensorBase *a, TensorBase *b, TensorBase *out)
{
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
