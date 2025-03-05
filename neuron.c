#include "neuron.h"

Neuron *neuron_create(int input_size, Activator *activator) {
    Neuron *neuron = (Neuron *)malloc(sizeof(Neuron));
    neuron->weights = vector_create(input_size);
    neuron->bias = 0;
    neuron->activator = activator;
    vector_randomize(neuron->weights, 0.0, 0.1);
    neuron->linear_output = 0;
    return neuron;
}

void neuron_free(Neuron *neuron) {
    vector_free(neuron->weights);
    free(neuron);
}

float neuron_activate(Neuron *neuron, Vector *input) {
    float sum = vector_dot(neuron->weights, input);
    sum += neuron->bias;
    neuron->linear_output = sum;
    return neuron->activator->activate(sum);
}