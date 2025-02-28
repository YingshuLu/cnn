#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include "operator.h"
#include "tensor.h"

void _tensor_destory(void *obj) {
    Tensor *tensor = (Tensor *)obj;
    for (int i = 0; i < tensor->depth; i++) {
        matrix_free(tensor->matrices[i]);
    }
    free(tensor->matrices);
    free(tensor);
}

Tensor *tensor_create_empty(int rows, int cols, int depth){
    Tensor *tensor = malloc(sizeof(Tensor));
    tensor->rows = rows;
    tensor->cols = cols;
    tensor->depth = depth;
    tensor->matrices = malloc(depth * sizeof(Matrix *));
    refer_init(&tensor->refer);
    return tensor;
}

Tensor *tensor_create(int rows, int cols, int depth) {
    Tensor *tensor = tensor_create_empty(rows, cols, depth);
    for (int i = 0; i < depth; i++) {
        tensor->matrices[i] = matrix_create(rows, cols);
    }
    return tensor;
}

void tensor_free(Tensor *tensor) {
    unrefer(tensor, _tensor_destory);
}

Tensor *tensor_create_from_array(float *data, int rows, int cols, int depth) {
    Tensor *tensor = tensor_create(rows, cols, depth);
    for (int d = 0; d < depth; d++) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                tensor_set(tensor, i, j, d, data[d * rows * cols + i * cols + j]);
            }
        }
    }
    return tensor;
}

Tensor *tensor_copy(Tensor *tensor) {
    Tensor *copy = tensor_create_empty(tensor->rows, tensor->cols, tensor->depth);
    for (int d = 0; d < tensor->depth; d++) {
        copy->matrices[d] = matrix_copy(tensor->matrices[d]);
    }
    return copy;
}

float tensor_get(Tensor *tensor, int row, int col, int dep) {
    return matrix_get(tensor->matrices[dep], row, col);
}

void tensor_set(Tensor *tensor, int row, int col, int dep, float value) {
    matrix_set(tensor->matrices[dep], row, col, value);
}

void tensor_assert_dimensions(Tensor *tensor, Tensor *other) {
    assert(tensor->rows == other->rows && 
    tensor->cols == other->cols && 
    tensor->depth == other->depth);
}

void tensor_tensor_operation(Tensor *tensor, Tensor *other, float (*operation)(float, float)) {
    tensor_assert_dimensions(tensor, other);

    for (int d = 0; d < tensor->depth; d++) {
        for (int i = 0; i < tensor->rows; i++) {
            for (int j = 0; j < tensor->cols; j++) {
                tensor_set(tensor, i, j, d, operation(tensor_get(tensor, i, j, d), tensor_get(other, i, j, d)));
            }
        }
    }
}

void _tensor_value_operation(Tensor *tensor, float value, float (*operation)(float, float)) {
    for (int d = 0; d < tensor->depth; d++) {
        for (int i = 0; i < tensor->rows; i++) {
            for (int j = 0; j < tensor->cols; j++) {
                tensor_set(tensor, i, j, d, operation(tensor_get(tensor, i, j, d), value));
            }
        }
    }
}

void tensor_apply(Tensor *tensor, float (*func)(float)) {
    for (int d = 0; d < tensor->depth; d++) {
        for (int i = 0; i < tensor->rows; i++) {
            for (int j = 0; j < tensor->cols; j++) {
                tensor_set(tensor, i, j, d, func(tensor_get(tensor, i, j, d)));
            }
        }
    }
}

void tensor_add_value(Tensor *tensor, float value) {
    _tensor_value_operation(tensor, value, float_add);
}

void tensor_sub_value(Tensor *tensor, float value) {
    _tensor_value_operation(tensor, value, float_sub);
}

void tensor_mul_value(Tensor *tensor, float value) {
    _tensor_value_operation(tensor, value, float_mul);
}

void tensor_div_value(Tensor *tensor, float value) {
    _tensor_value_operation(tensor, value, float_div);
}

void tensor_mod_value(Tensor *tensor, float value) {
    _tensor_value_operation(tensor, value, float_mod);
}

void tensor_pow_value(Tensor *tensor, float value) {
    _tensor_value_operation(tensor, value, float_pow);
}

void tensor_fill_value(Tensor *tensor, float value) {
    _tensor_value_operation(tensor, value, float_fill);
}

void tensor_concat_copy(Tensor *input, Tensor *target) {
    assert(input->rows == target->rows && input->cols == target->cols);

    input->depth += target->depth;
    input->matrices = realloc(input->matrices, input->depth * sizeof(Matrix *));
    for (int i = 0; i < target->depth; i++) {
        input->matrices[input->depth - target->depth + i] = matrix_copy(target->matrices[i]);
    }
}

void tensor_concat_refer(Tensor *input, Tensor *target) {
    assert(input->rows == target->rows && input->cols == target->cols);

    input->depth += target->depth;
    input->matrices = realloc(input->matrices, input->depth * sizeof(Matrix *));
    for (int i = 0; i < target->depth; i++) {
        input->matrices[input->depth - target->depth + i] = matrix_refer(target->matrices[i]);
    }
}