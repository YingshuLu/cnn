#include "tensor_network.h"

TensorNetwork *tensor_network_create() {
    TensorNetwork *network = (TensorNetwork *)malloc(sizeof(TensorNetwork));
    network->layers = (TensorLayer **)calloc(4, sizeof(TensorLayer *));
    network->count = 0;
    network->capacity = 4;
    return network;
}

void tensor_network_add_layer(TensorNetwork *network, TensorLayer *layer) {
    if (network->count >= network->capacity) {
        network->capacity *= 2;
        network->layers = (TensorLayer **)realloc(network->layers, network->capacity * sizeof(TensorLayer *));
    }
    network->layers[network->count++] = layer;
}

Tensor *tensor_network_forward(TensorNetwork *network, Tensor *input) { 
    Tensor *tensor = input;
    for (int i = 0; i < network->count; i++) {
        tensor = network->layers[i]->forward(network->layers[i], tensor);
    }
    return tensor;
}

Tensor *tensor_network_backward(TensorNetwork *network, Tensor *gradient) {
    Tensor *tensor = gradient;
    for (int i = network->count - 1; i >= 0; i--) {
        tensor = network->layers[i]->backward(network->layers[i], tensor);
    }
    return tensor;
}

void tensor_network_free(TensorNetwork *network) {
    for (int i = 0; i < network->count; i++) {
        network->layers[i]->free(network->layers[i]);
    }
    free(network->layers);
    free(network);
}