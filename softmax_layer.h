#ifndef SOFTMAX_LAYER_H
#define SOFTMAX_LAYER_H

#include "layer.h"
#include "vector.h"

typedef struct SoftmaxLayer {
    Layer layer;
    Vector *input;
} SoftmaxLayer;

SoftmaxLayer *softmax_layer_create();
void softmax_layer_free(SoftmaxLayer *layer);
Vector *softmax_layer_forward(SoftmaxLayer *layer, Vector *input);
Vector *softmax_layer_backward(SoftmaxLayer *layer, Vector *gradient);

#endif
