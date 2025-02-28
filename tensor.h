#ifndef TENSOR_H
#define TENSOR_H

typedef struct Tensor {
    int rows;
    int cols;
    int depth;
    float *data;
} Tensor;

Tensor *tensor_create(int rows, int cols, int depth);
void tensor_free(Tensor *tensor);
Tensor *tensor_create_from_array(float *data, int rows, int cols, int depth);
float tensor_get(Tensor *tensor, int row, int col, int depth);
void tensor_set(Tensor *tensor, int row, int col, int depth, float value);
void tensor_add_value(Tensor *tensor, float value);
void tensor_sub_value(Tensor *tensor, float value);
void tensor_mul_value(Tensor *tensor, float value);
void tensor_div_value(Tensor *tensor, float value);
void tensor_mod_value(Tensor *tensor, float value);
void tensor_pow_value(Tensor *tensor, float value);
void tensor_fill_value(Tensor *tensor, float value);

#endif