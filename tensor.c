#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include "operator.h"
#include "tensor.h"

Tensor *tensor_create(int rows, int cols, int depth) {
    Tensor *tensor = malloc(sizeof(Tensor));
    tensor->rows = rows;
    tensor->cols = cols;
    tensor->depth = depth;
    tensor->data = calloc(rows * cols * depth, sizeof(float));
    return tensor;
}

void tensor_free(Tensor *tensor) {
    free(tensor->data);
    free(tensor);
}

Tensor *tensor_create_from_array(float *data, int rows, int cols, int depth) {
    Tensor *tensor = tensor_create(rows, cols, depth);
    memcpy(tensor->data, data, rows * cols * depth * sizeof(float));
    return tensor;
}

float tensor_get(Tensor *tensor, int row, int col, int depth) {
    int base_index = depth * tensor->rows * tensor->cols;
    return tensor->data[base_index + row * tensor->cols + col];
}

void tensor_set(Tensor *tensor, int row, int col, int depth, float value) {
    int base_index = depth * tensor->rows * tensor->cols;
    tensor->data[base_index + row * tensor->cols + col] = value;
}

void tensor_assert_dimensions(Tensor *tensor, Tensor *other) {
    assert(tensor->rows == other->rows && 
    tensor->cols == other->cols && 
    tensor->depth == other->depth);
}

void tensor_tensor_operation(Tensor *tensor, Tensor *other, float (*operation)(float, float)) {
    tensor_assert_dimensions(tensor, other);

    for (int i = 0; i < tensor->rows * tensor->cols * tensor->depth; i++) {
        tensor->data[i] = operation(tensor->data[i], other->data[i]);
    }
}

void _tensor_value_operation(Tensor *tensor, float value, float (*operation)(float, float)) {
    for (int i = 0; i < tensor->rows * tensor->cols * tensor->depth; i++) {
        tensor->data[i] = operation(tensor->data[i], value);
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
