#ifndef TENSOR_LAYER_H
#define TENSOR_LAYER_H

#include "tensor.h"

typedef struct TensorLayer {
    Tensor *(*forward)(void *layer, Tensor *input);
    Tensor *(*backward)(void *layer, Tensor *grad_output);
    void (*free)(void *layer);
} TensorLayer;

#endif
