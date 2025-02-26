#ifndef NEURON_H
#define NEURON_H

#include "vector.h"
#include "activation.h"

typedef struct Neuron {
    Vector *weights;
    float bias;
    Activation *activation;
    float linear_output;
} Neuron;

Neuron *neuron_create(int input_size, Activation *activation);
void neuron_destroy(Neuron *neuron);
float neuron_activate(Neuron *neuron, Vector *input);

#endif
