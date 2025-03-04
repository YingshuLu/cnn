#ifndef MAX_POOL_LAYER_H
#define MAX_POOL_LAYER_H

#include "tensor.h"

typedef struct MaxPoolLayer {
    int pool_size;
    int stride;
} MaxPoolLayer;

MaxPoolLayer *max_pool_layer_create(int pool_size, int stride);
void max_pool_layer_free(MaxPoolLayer *layer);
Tensor *max_pool_layer_forward(MaxPoolLayer *layer, Tensor *input);
Tensor *max_pool_layer_backward(MaxPoolLayer *layer, Tensor *input, Tensor *gradient);
Tensor *max_pool_layer_gradient(MaxPoolLayer *layer, Tensor *input, Tensor *gradient);


#endif