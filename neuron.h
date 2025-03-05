#ifndef NEURON_H
#define NEURON_H

#include "vector.h"
#include "activator.h"

typedef struct Neuron {
    Vector *weights;
    float bias;
    Activator *activator;
    float linear_output;
} Neuron;

Neuron *neuron_create(int input_size, Activator *activator);
void neuron_free(Neuron *neuron);
float neuron_activate(Neuron *neuron, Vector *input);

#endif
