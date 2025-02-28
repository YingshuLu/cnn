#ifndef NEURON_LAYER_H
#define NEURON_LAYER_H

#include "layer.h"

typedef struct LayerNeuron {
    Layer layer;
    Neuron **neurons;
    int neurons_size;
    float learning_rate;
    int input_size;
    Vector* input;
    Vector* output;
} LayerNeuron;

LayerNeuron *layer_neuron_create(int size, int input_size, Activation *activation, float learning_rate);
void layer_neuron_free(LayerNeuron *layer);

#endif
