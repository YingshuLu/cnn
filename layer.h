#ifndef LAYER_H
#define LAYER_H

#include "neuron.h"

typedef struct Layer {
    Vector *(*forward)(void *layer, Vector *input);
    Vector *(*backward)(void *layer, Vector *gradient);
    void (*free)(void *layer);
} Layer;

#endif
