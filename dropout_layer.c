#include <math.h>
#include "dropout_layer.h"

Vector *_dropout_layer_forward(void *layer, Vector *input) {
    DropoutLayer *dropout_layer = (DropoutLayer*) layer;
    return dropout_layer_forward(dropout_layer, input);
}

Vector *_dropout_layer_backward(void *layer, Vector *gradient) {
    DropoutLayer *dropout_layer = (DropoutLayer *) layer;
    return dropout_layer_backward(layer, gradient);
}

void _dropout_layer_free(void *layer) {
    DropoutLayer *dropout_layer = (DropoutLayer *)layer;
    free(dropout_layer);
}

DropoutLayer *dropout_layer_create(float dropout_rate) {
    DropoutLayer *layer = (DropoutLayer *)malloc(sizeof(DropoutLayer));
    layer->dropout_rate = dropout_rate;
    layer->layer.forward = _dropout_layer_forward;
    layer->layer.backward = _dropout_layer_backward;
    layer->layer.free = _dropout_layer_free;
    return layer;
}

void dropoutLayer_free(DropoutLayer *layer) {
    free(layer);
}

Vector *dropout_layer_forward(DropoutLayer *layer, Vector *input) {
    layer->mask = vector_create(input->size);

    int base_line = layer->dropout_rate * 100;
    float scale = 1 / (1 - layer->dropout_rate);
    Vector *output = vector_create(input->size);
    for (int i = 0; i < input->size; i++) {
        if (rand() % 100 > base_line) {
            layer->mask->data[i] = scale;
            output->data[i] = input->data[i];
        } else {
            layer->mask->data[i] = 0.0f;
            output->data[i] = 0.0f;
        }
    }
    return output;
}

Vector *dropout_layer_backward(DropoutLayer *layer, Vector *gradient) {
    Vector *new_gradient = layer->mask;
    layer->mask = 0;
    for (int i = 0; i < gradient->size; i++) {
        new_gradient->data[i] *= gradient->data[i];
    }
    return new_gradient;
}
