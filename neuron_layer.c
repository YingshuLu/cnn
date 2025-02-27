#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <math.h>
#include "neuron_layer.h"

/*
Define: network has 3 layers:
    layer 1:
        X = [x1, x2, x3]
    
    layer 2:
        M = F(X) = W1 * X + B1
        N = L(M) = LERU(M)

    layer 3:
        P = G(N) = W2 * N + B2
        Y = H(P) = SIGMOID(P)

so backward gradient:

    layer 3:
        dY/dW2 = dY/dP * dP/dW2 = dSIGMOD(P)/dP * N

        dY/dB2 = dY/dP * dP/dB2 = dSIGMOD(P)/dP * 1

    layer 2:
        dY/dW1 = dY/dP * dP/dN * dN/dM * dM/dW1 = Sum(dSIGMOD(P)/dP * W2) * dLERU(M)/dM * X

        dY/dB1 = dY/dP * dP/dN * dN/dM * dM/dB1 = Sum(dSIGMOD(P)/dP * W2) * dLERU(M)/dM * 1
*/

void layer_neuron_update_input(LayerNeuron *layer, Vector *input) {
    vector_destroy(layer->input);
    refer(input);
    layer->input = input;
}

void layer_neuron_update_output(LayerNeuron *layer, Vector *output) {
    vector_destroy(layer->output);
    refer(output);
    layer->output = output;
}

Vector *layer_neuron_forward(void *layer_base, Vector *input) {
    LayerNeuron *layer = (LayerNeuron *)layer_base;
    assert(input->size == layer->input_size);
    layer_neuron_update_input(layer, input);
    Vector* output = vector_create(layer->neurons_size);
    for (int i = 0; i < layer->neurons_size; i++) {
        output->data[i] = neuron_activate(layer->neurons[i], input);
    }
    layer_neuron_update_output(layer, output);
    return output;
}

Vector *layer_neuron_backward(void *layer_base, Vector *gradient) {
    LayerNeuron *layer = (LayerNeuron *)layer_base;
    assert(layer->neurons_size == gradient->size);

    Vector *new_gradient = vector_create(layer->input_size);
    vector_fill(new_gradient, 0);

    Vector *input = layer->input;
    float learning_rate = layer->learning_rate;

    Neuron *neuron;
    float loss_value, delta;
    for (int i = 0; i < layer->neurons_size; i++) {
        neuron = layer->neurons[i];
        loss_value = gradient->data[i];
        float error = neuron->activation->derivate(neuron->linear_output);
        delta = error * loss_value;
        if (fabs(delta) > 1.0) {
            delta = delta > 0 ? 1.0 : -1.0;
        }

        // calculate the previous layer's gradient
        Vector* neuron_loss = vector_copy(neuron->weights);
        vector_mul_value(neuron_loss, delta);
        vector_add(new_gradient, neuron_loss);
        vector_destroy(neuron_loss);

        // update the weights
        Vector *input_vector = vector_copy(input);
        vector_mul_value(input_vector, learning_rate * delta);
        vector_sub(neuron->weights, input_vector);
        vector_destroy(input_vector);
        neuron->bias -= learning_rate * delta;
    }
    vector_destroy(gradient);
    return new_gradient;
}

LayerNeuron *layer_neuron_create(int size, int input_size, Activation *activation, float learning_rate) {
    LayerNeuron *layer = (LayerNeuron *)malloc(sizeof(LayerNeuron));
    layer->neurons = (Neuron **)malloc(size * sizeof(Neuron *));
    layer->neurons_size = size;
    layer->input_size = input_size;
    layer->learning_rate = learning_rate;
    for (int i = 0; i < size; i++) {
        layer->neurons[i] = neuron_create(input_size, activation);
    }

    layer->layer.forward = layer_neuron_forward;
    layer->layer.backward = layer_neuron_backward;
    return layer;
}

void layer_neuron_destroy(LayerNeuron *layer) {
    for (int i = 0; i < layer->neurons_size; i++) {
        neuron_destroy(layer->neurons[i]);
    }
    free(layer->neurons);
    vector_destroy(layer->input);
    vector_destroy(layer->output);
    free(layer);
}



