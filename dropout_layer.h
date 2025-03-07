#ifndef DROPOUT_H
#define DROPOUT_H

#include "layer.h"
#include "vector.h"

typedef struct DropoutLayer {
    Layer layer;
    float dropout_rate;
    Vector *mask;
} DropoutLayer;

DropoutLayer *dropout_layer_create(float dropout_rate);
void dropout_layer_free(DropoutLayer *layer);
Vector *dropout_layer_forward(DropoutLayer *layer, Vector *input);
Vector *dropout_layer_backward(DropoutLayer *layer, Vector *output);

#endif