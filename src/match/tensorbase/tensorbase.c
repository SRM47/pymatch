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
 *                        Utility                        *
 *********************************************************/

static inline int TensorBase_is_singleton(TensorBase *t)
{
    return t->ndim == 0;
}

static inline int TensorBase_compare_shape(ShapeArray a_shape, ShapeArray b_shape)
{
    return memcmp(a_shape, b_shape, MAX_RANK * sizeof(long)) == 0;
}

static int TensorBase_create_empty_like(TensorBase *in, TensorBase *out)
{
    // Assumes out->data doesn't point to any alocated memory.
    if (in == NULL || out == NULL)
    {
        return -1; // Invalid input or output tensor
    }

    memcpy(out, in, sizeof(TensorBase));

    if (!TensorBase_is_singleton(in))
    {
        out->data = (scalar *)malloc(in->numel * sizeof(scalar));
        if (out->data == NULL)
        {
            return -1;
        }
    }

    return 0;
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
    // If TensorBase is singleton, the data pointer will hold the value
    // instead of pointing to a one element array.
    if (ndim != 0)
    {
        td->data = (scalar *)malloc(td->numel * sizeof(scalar));
        if (td->data == NULL)
        {
            // Memory error.
            return -2;
        }
    }
    else
    {
        // If singleton, initialize the pointer to be null.
        // TensorBase_create_tensor_like will not free this then,
        td->data = NULL;
    }

    return 0;
}

void TensorBase_dealloc(TensorBase *td)
{
    if (td == NULL)
    {
        return;
    }

    // Only free the pointer if not singleton.
    // Sington tensor structs do not point to address on heap,
    // rather directly store data in the pointer variable.
    // also assumes only one pointer to data. will not implementreference counting if we start to share memory between tensors
    if (td->data != NULL && !TensorBase_is_singleton(td))
    {
        free(td->data);
    }
    // Use memset to zero out shape and strides arrays safely.
    memset(td, 0, sizeof(TensorBase));
}

/*********************************************************
 *                     String Methods                    *
 *********************************************************/

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

/*********************************************************
 *                     Braodcasting                      *
 *********************************************************/

static int TensorBase_get_broadcast_shape(ShapeArray a_shape, long a_ndim, ShapeArray b_shape, long b_ndim, ShapeArray broadcast_shape, long *broadcast_ndim)
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

static inline void apply_binop(BinaryScalarOperation binop, scalar a, scalar b, scalar *result)
{
    switch (binop)
    {
    case SCALAR_ADD:
        *result = a + b;
        break;
    case SCALAR_SUB:
        *result = a - b;
        break;
    case SCALAR_MULT:
        *result = a * b;
        break;
    case SCALAR_FLOORDIV:
        *result = floor(a / b);
        break;
    case SCALAR_TRUEDIV:
        *result = a / b;
        break;
    case SCALAR_POWER:
        *result = pow(a, b);
        break;
    default:
        break;
    }
}

int TensorBase_binary_op_tensorbase_scalar(TensorBase *a, scalar s, TensorBase *out, BinaryScalarOperation binop)
{
    if (TensorBase_create_empty_like(a, out) == -1)
    {
        return -1;
    }
    if (TensorBase_is_singleton(a))
    {
        // Copy the bits of a->data into a_value.
        scalar a_value;
        memcpy(&a_value, &(a->data), sizeof(scalar));
        // Compute the result of the operator.
        scalar result;
        apply_binop(binop, a_value, s, &result);
        // Copy the bits of result into out->data.
        memcpy(&(out->data), &result, sizeof(scalar));
    }
    else
    {
        for (long i = 0; i < out->numel; i++)
        {
            apply_binop(binop, a->data[i], s, out->data + i);
        }
    }
    return 0;
}

int TensorBase_binary_op_scalar_tensorbase(TensorBase *a, scalar s, TensorBase *out, BinaryScalarOperation binop)
{
    if (TensorBase_create_empty_like(a, out) == -1)
    {
        return -1;
    }
    if (TensorBase_is_singleton(a))
    {
        // Copy the bits of a->data into a_value.
        scalar a_value;
        memcpy(&a_value, &(a->data), sizeof(scalar));
        // Compute the result of the operator.
        scalar result;
        apply_binop(binop, s, a_value, &result);
        // Copy the bits of result into out->data.
        memcpy(&(out->data), &result, sizeof(scalar));
    }
    else
    {
        for (long i = 0; i < out->numel; i++)
        {
            apply_binop(binop, s, a->data[i], out->data + i);
        }
    }
    return 0;
}

int TensorBase_binary_op_tensorbase_tensorbase(TensorBase *a, TensorBase *b, TensorBase *out, BinaryScalarOperation binop)
{
    // Check if a is a singleton.
    if (TensorBase_is_singleton(a))
    {
        scalar s;
        memcpy(&s, &(a->data), sizeof(scalar));
        return TensorBase_binary_op_scalar_tensorbase(b, s, out, binop);
    }

    // Check if b is a singleton.
    if (TensorBase_is_singleton(b))
    {
        scalar s;
        memcpy(&s, &(b->data), sizeof(scalar));
        return TensorBase_binary_op_tensorbase_scalar(a, s, out, binop);
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
            apply_binop(binop, a->data[i], b->data[i], out->data + i);
        }
    }
    else
    {
        // They don't have the same shape must *attempt to* broadcast.
        ShapeArray broadcasted_tensor_shape;
        long broadcasted_tensor_ndim;
        if (TensorBase_get_broadcast_shape(a->shape, a->ndim, b->shape, b->ndim, broadcasted_tensor_shape, &broadcasted_tensor_ndim) == -1)
        {
            // Incompatible broadcasting shapes.
            return -1;
        }

        if (TensorBase_init(out, broadcasted_tensor_shape, broadcasted_tensor_ndim) == -1)
        {
            return -1;
        }
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

            apply_binop(binop, a->data[a_data_index], b->data[b_data_index], out->data + broadcasted_data_index);
        }
    }

    return 0;
}

static inline void apply_uop(UnaryScalarOperation uop, scalar a, scalar *result)
{
    switch (uop)
    {
    case SCALAR_NEGATIVE:
        *result = -a;
        break;
    case SCALAR_ABSOLUTE:
        *result = fabs(a);
        break;
    case SCALAR_COS:
        *result = cos(a);
        break;
    case SCALAR_SIN:
        *result = sin(a);
        break;
    case SCALAR_TAN:
        *result = tan(a);
        break;
    case SCALAR_TANH:
        *result = tanh(a);
        break;
    case SCALAR_LOG:
        *result = log(a);
        break;
    case SCALAR_EXP:
        *result = exp(a);
        break;
    case SCALAR_SIGMOID:
        *result = 1.0 / (1.0 + exp(-a));
        break;
    default:
        break;
    }
}

int TensorBase_unary_op_inplace(TensorBase *in, UnaryScalarOperation uop)
{
    if (in->data == NULL)
    {
        return -1;
    }
    if (TensorBase_is_singleton(in))
    {
        scalar in_value;
        memcpy(&in_value, &(in->data), sizeof(scalar));

        scalar result;
        apply_uop(uop, in_value, &result);

        memcpy(&(in->data), &result, sizeof(scalar));
    }
    else
    {
        for (long i = 0; i < in->numel; i++)
        {
            apply_uop(uop, in->data[i], in->data + i);
        }
    }
    return 0;
}

int TensorBase_unary_op(TensorBase *in, TensorBase *out, UnaryScalarOperation uop)
{
    if (TensorBase_create_empty_like(in, out) == -1)
    {
        return -1;
    }
    if (TensorBase_is_singleton(in))
    {
        scalar in_value;
        memcpy(&in_value, &(in->data), sizeof(scalar));

        scalar result;
        apply_uop(uop, in_value, &result);

        memcpy(&(out->data), &result, sizeof(scalar));
    }
    else
    {
        for (long i = 0; i < out->numel; i++)
        {
            apply_uop(uop, in->data[i], out->data + i);
        }
    }
    return 0;
}

// TODO: Implement
int TensorBase_initialize_for_matrix_multiplication(TensorBase *a, TensorBase *b, TensorBase *out)
{
    if (a == NULL || b == NULL || out == NULL)
    {
        return -1;
    }

    if (TensorBase_is_singleton(a) || TensorBase_is_singleton(b))
    {
        return -1;
    }

    ShapeArray shape;
    StrideArray strides;
    long ndim;
    long numel;

    if (a->ndim == 1 && b->ndim == 1)
    {
        if (a->numel != b->numel)
        {
            return -1;
        }
        ndim = 0;
        numel = 1;
    }
    else if (a->ndim == 1 && b->ndim == 2)
    {
        if (a->shape[0] != b->shape[0])
        {
            return -1;
        }
        shape[0] = b->shape[1];
        strides[0] = 1;
        ndim = 1;
        numel = b->shape[1];
    }
    else if (a->ndim == 2 && b->ndim == 1)
    {
        if (a->shape[1] != b->shape[0])
        {
            return -1;
        }
        shape[0] = a->shape[0];
        strides[0] = 1;
        ndim = 1;
        numel = a->shape[0];
    }
    else if (a->ndim == 2 && b->ndim == 2)
    {
        if (a->shape[1] != b->shape[0])
        {
            return -1;
        }
        shape[0] = a->shape[0];
        shape[1] = b->shape[1];
        strides[0] = b->shape[1];
        strides[1] = 1;
        ndim = 2;
        numel = a->shape[0] * b->shape[1];
    }
    else
    { // are shapes compatible.
        long matrix_dims_a = a->ndim > 1 ? 2 : 1;
        long matrix_dims_b = b->ndim > 1 ? 2 : 1;
        long batch_dims_a = a->ndim - matrix_dims_a; // Number of non matrix dimensions in a
        long batch_dims_b = b->ndim - matrix_dims_b; // Number of non-matrix dimensions in b

        // matrix dimensions begin at batch_dims + 1
        // if a.shape = [2,3,4,5,6]
        // batch dims are [2,3,4] (batch_dim_a = 3)
        // so matrix dims are [5,6], or shape[batch_dim_a] and shape[batch_dim_a+1].
        if (a->shape[batch_dims_a + 1] != b->shape[batch_dims_b])
        {
            return -2;
        }

        long non_matrix_dims;
        // Broadcast the non-matrx dimensions.
        if (TensorBase_get_broadcast_shape(a->shape, batch_dims_a, b->shape, batch_dims_b, shape, &non_matrix_dims) == -1)
        {
            return -2;
        }

        ndim = non_matrix_dims;
        if (matrix_dims_a == 1)
        {
            shape[non_matrix_dims] = b->shape[batch_dims_b + 1];
            ndim += 1;
        }
        else if (matrix_dims_b == 1)
        {
            shape[non_matrix_dims] = a->shape[batch_dims_a];
            ndim += 1;
        }
        else
        {
            shape[non_matrix_dims] = a->shape[batch_dims_a];
            shape[non_matrix_dims + 1] = b->shape[batch_dims_b + 1];
            ndim += 2;
        }

        // Calculate numel and strides
        // Calculate Tensorbase shape and number of elements.
        long numel_for_stride = 1;
        numel = 1;
        for (int i = 0; i < ndim; i++)
        {
            long dim = shape[i];
            numel *= dim;
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
            strides[i] = stride;
        }
    }

    for (long i = ndim; i < MAX_RANK; i++)
    {
        shape[i] = -1;
        strides[i] = 0;
    }
    memcpy(&out->shape, &shape, sizeof(scalar) * MAX_RANK);
    memcpy(&out->strides, &strides, sizeof(scalar) * MAX_RANK);
    out->numel = numel;
    out->ndim = ndim;
    if (ndim > 0)
    {
        out->data = (scalar *)malloc(numel * sizeof(scalar));
    }



    return 0;
}

int matrix_multiply_2d(scalar *A, scalar *B, long n, long l, long m, scalar *out)
{
    // Assumes A is a n x l matrix
    // Assumes B is a l x m matrix
    // out = A@B will be a n x m matrix
    // Assumes out is already allocated
    for (long i = 0; i < n; i++)
    {
        for (long j = 0; j < m; j++)
        {
            scalar sum = 0;
            for (long k = 0; k < l; k++)
            {
                sum += A[i * l + k] * B[k * m + j];
            }
            out[i * m + j] = sum;
        }
    }

    return 0;
}
// TODO: Implement
int TensorBase_matrix_multiply(TensorBase *a, TensorBase *b, TensorBase *out)
{
    if (a == NULL || b == NULL || out == NULL)
    {
        return -1;
    }

    int status = TensorBase_initialize_for_matrix_multiplication(a, b, out);
    if (status < 0)
    {
        return status;
    }

    if (a->ndim == 1 && b->ndim == 1)
    {
        scalar sum = 0;
        for (long i = 0; i < a->numel; i++)
        {
            sum += a->data[i] * b->data[i];
        }
        memcpy(&out->data, &sum, sizeof(scalar));
    }
    else if (a->ndim == 1 && b->ndim == 2)
    {
        return matrix_multiply_2d(a->data, b->data, 1, a->shape[0], b->shape[1], out->data);
    }
    else if (a->ndim == 2 && b->ndim == 1)
    {
        return matrix_multiply_2d(a->data, b->data, a->shape[0], b->shape[0], 1, out->data);
    }
    else if (a->ndim == 2 && b->ndim == 2)
    {
        return matrix_multiply_2d(a->data, b->data, a->shape[0], a->shape[1], b->shape[1], out->data);
    }
    else
    {
        return 0;
    }
}

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

int TensorBase_reshape_inplace(TensorBase *in, ShapeArray shape, long ndim)
{
    // Does not do validation of shape array
    // assumes ndim is the rank of shape
    if (in == NULL)
    {
        return -1;
    }

    if (ndim > MAX_RANK)
    {
        return -4;
    }

    long numel_for_stride = 1;
    long numel = 1;
    printf("%ld ", ndim);
    for (int i = 0; i < ndim; i++)
    {
        long dim = shape[i];
        printf("%ld ", dim);
        if (dim < 0)
        {
            // All dimensions must be >= 0 (-1 indicates no-dimension).
            return -2;
        }
        numel *= dim;
        if (dim > 0)
        {
            numel_for_stride *= dim;
        }
    }

    if (numel != in->numel)
    {
        return -3;
    }

    // Calculate if the input tensor is singleton before changing internal metadata.
    int is_singleton = TensorBase_is_singleton(in);

    // Calculate Tensorbase strides.
    in->ndim = ndim;
    long stride = numel_for_stride;
    long i = 0;
    for (; i < ndim; i++)
    {
        long dim = shape[i];
        if (dim > 0)
        {
            stride /= dim;
        }
        in->strides[i] = stride;
        in->shape[i] = dim;
    }
    for (; i < MAX_RANK; i++)
    {
        in->strides[i] = 0;
        in->shape[i] = -1;
    }

    // Must handle special case for singleton objects.
    // Singleton -> nd. must allocate memory for the one element in the nd array
    if (is_singleton)
    {
        if (ndim > 0)
        {
            // Allocate the new memory.
            scalar *new_data_region = (scalar *)malloc(1 * sizeof(scalar));
            if (new_data_region == NULL)
            {
                return -1;
            }
            // Copy the bits of the original scalar into the first position of the new_data_region.
            memcpy(new_data_region, &(in->data), sizeof(scalar));
            // Assign the data point the new region in memory.
            in->data = new_data_region;
        }
    }
    // nd -> singleton. must free memory and place the singleton value in the pointer.
    else
    {
        // By this point, if ndim is 0, the bew shape will be a singleton and all the checks wouldv'e been verified.
        // this is just copying the data.
        if (ndim == 0)
        {
            // also note that the fact that the singleton data isnt just a one element array meas that it's hard now to share data
            // we lose that functionality for better locality and fewer allocations.
            // Get the single value from the nd tensor
            scalar value = *in->data;
            // Free the memory
            free(in->data);
            // Copy the memory bits from the scalar into the in->data pointer
            memcpy(&(in->data), &value, sizeof(scalar));
        }
    }

    return 0;
}
int TensorBase_reshape(TensorBase *in, TensorBase *out, ShapeArray shape, long ndim)
{
    // Does not do validation of shape array
    // assumes ndim is the rank of shape
    if (in == NULL || out == NULL)
    {
        return -1;
    }

    if (ndim > MAX_RANK)
    {
        return -4;
    }

    // will not share data memory (data pointers point to the same data) or else we'd have to implement reference counting of the data element
    // well call this a limitation of the system
    if (TensorBase_create_empty_like(in, out) < 0)
    {
        return -1;
    }

    // must also copy the data over if not a single object.
    if (!TensorBase_is_singleton(in))
    {
        memcpy(out->data, in->data, out->numel * sizeof(scalar));
    }

    return TensorBase_reshape_inplace(out, shape, ndim);
}

int TensorBase_fill_(TensorBase *in, scalar fill_value)
{
    if (TensorBase_is_singleton(in))
    {
        memcpy(&(in->data), &fill_value, sizeof(scalar));
        return 0;
    }

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

    if (TensorBase_is_singleton(t))
    {
        memcpy(item, &(t->data), sizeof(scalar));
    }
    else
    {
        *item = *(t->data);
    }

    return 0;
}
