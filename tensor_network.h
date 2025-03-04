#ifndef TENSOR_NETWORK_H
#define TENSOR_NETWORK_H

#include "tensor_layer.h"

typedef struct TensorNetwork {
    TensorLayer **layers;
    int count;
    int capacity;
} TensorNetwork;

TensorNetwork *tensor_network_create();
void tensor_network_add_layer(TensorNetwork *network, TensorLayer *layer);
Tensor *tensor_network_forward(TensorNetwork *network, Tensor *input);
Tensor *tensor_network_backward(TensorNetwork *network, Tensor *gradient);
void tensor_network_free(TensorNetwork *network);

#endif
