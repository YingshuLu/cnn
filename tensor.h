#ifndef TENSOR_H
#define TENSOR_H

#include "matrix.h"
#include "refer.h"

#define tensor_refer(tensor) (refer(&(tensor)->refer), (tensor))
#define tensor_unrefer(tensor) tensor_free(tensor)

typedef struct Tensor {
    Refer refer;
    int rows;
    int cols;
    int depth;
    Matrix **matrices;
} Tensor;

Tensor *tensor_create(int rows, int cols, int depth);
void tensor_free(Tensor *tensor);
Tensor *tensor_create_from_array(float *data, int rows, int cols, int depth);
Tensor *tensor_copy(Tensor *tensor);
float tensor_get(Tensor *tensor, int row, int col, int depth);
void tensor_set(Tensor *tensor, int row, int col, int depth, float value);
void tensor_apply(Tensor *tensor, float (*func)(float));
void tensor_add_value(Tensor *tensor, float value);
void tensor_sub_value(Tensor *tensor, float value);
void tensor_mul_value(Tensor *tensor, float value);
void tensor_div_value(Tensor *tensor, float value);
void tensor_mod_value(Tensor *tensor, float value);
void tensor_pow_value(Tensor *tensor, float value);
void tensor_fill_value(Tensor *tensor, float value);
float tensor_sum(Tensor *tensor);
void tensor_concat_copy(Tensor *input, Tensor *target);
void tensor_concat_refer(Tensor *input, Tensor *target);

void tensor_add(Tensor *tensor, Tensor *other);
void tensor_sub(Tensor *tensor, Tensor *other);
void tensor_mul(Tensor *tensor, Tensor *other);
void tensor_div(Tensor *tensor, Tensor *other);

Tensor *tensor_slice_copy(Tensor *tensor, int depth_start, int depth_end);
Tensor *tensor_slice_refer(Tensor *tensor, int depth_start, int depth_end);

#endif