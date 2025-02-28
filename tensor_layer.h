#ifndef TENSOR_LAYER_H
#define TENSOR_LAYER_H

#include "tensor.h"
typedef struct TensorLayer {
    Tensor *(*forward)(struct TensorLayer *layer, Tensor *input);
    Tensor *(*backward)(struct TensorLayer *layer, Tensor *grad_output);
} TensorLayer;

#endif
