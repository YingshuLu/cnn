#ifndef MAX_POOL_LAYER_H
#define MAX_POOL_LAYER_H

#include "tensor.h"
#include "tensor_layer.h"

typedef struct MaxPoolLayer {
    TensorLayer layer;
    int pool_size;
    int stride;
    Tensor *input_mask;
} MaxPoolLayer;

MaxPoolLayer *max_pool_layer_create(int pool_size, int stride);
void max_pool_layer_free(MaxPoolLayer *layer);
Tensor *max_pool_layer_forward(MaxPoolLayer *layer, Tensor *input);
Tensor *max_pool_layer_backward(MaxPoolLayer *layer, Tensor *gradient);

#endif