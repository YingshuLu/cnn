#ifndef NEURON_H
#define NEURON_H

#include "vector.h"
#include "activation.h"

typedef struct Neuron {
    Vector *weights;
    float bias;
    Activator *activation;
    float linear_output;
} Neuron;

Neuron *neuron_create(int input_size, Activator *activation);
void neuron_free(Neuron *neuron);
float neuron_activate(Neuron *neuron, Vector *input);

#endif
