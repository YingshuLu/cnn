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

void _layer_neuron_update_input(LayerNeuron *layer, Vector *input) {
    vector_unrefer(layer->input);
    layer->input = vector_refer(input);
}

void _layer_neuron_update_output(LayerNeuron *layer, Vector *output) {
    vector_unrefer(layer->output);
    layer->output = vector_refer(output);
}

Vector *_layer_neuron_forward(void *layer_base, Vector *input) {
    LayerNeuron *layer = (LayerNeuron *)layer_base;
    if (layer->layze) {
        layer->input_size = input->size;
        for (int i = 0; i < layer->neurons_size; i++) {
            layer->neurons[i] = neuron_create(layer->input_size, layer->activator);
        }
        layer->layze = 0;
    }

    assert(input->size == layer->input_size);
    _layer_neuron_update_input(layer, input);
    Vector* output = vector_create(layer->neurons_size);

    #pragma omp parallel for
    for (int i = 0; i < layer->neurons_size; i++) {
        output->data[i] = neuron_activate(layer->neurons[i], input);
    }
    
    _layer_neuron_update_output(layer, output);
    return output;
}

Vector *_layer_neuron_backward(void *layer_base, Vector *gradient) {
    LayerNeuron *layer = (LayerNeuron *)layer_base;
    assert(layer->neurons_size == gradient->size);

    Vector *new_gradient = vector_create(layer->input_size);
    vector_fill(new_gradient, 0);

    Vector *input = layer->input;
    float learning_rate = layer->learning_rate;

    #pragma omp parallel for
    for (int i = 0; i < layer->neurons_size; i++) {
        Neuron *neuron = layer->neurons[i];
        float loss_value = gradient->data[i];
        float error = neuron->activator->derivate(neuron->linear_output);
        float delta = error * loss_value;
        if (fabs(delta) > 1.0) {
            delta = delta > 0 ? 1.0 : -1.0;
        }

        // calculate the previous layer's gradient
        Vector* neuron_loss = vector_copy(neuron->weights);
        vector_mul_value(neuron_loss, delta);
        #pragma omp critical
        {
            vector_add(new_gradient, neuron_loss);
        }
        vector_free(neuron_loss);

        // update the weights
        Vector *input_vector = vector_copy(input);
        vector_mul_value(input_vector, learning_rate * delta);
        vector_sub(neuron->weights, input_vector);
        vector_free(input_vector);
        neuron->bias -= learning_rate * delta;
    }

    vector_free(gradient);
    return new_gradient;
}

void _layer_neuron_free(void *layer_base) {
    LayerNeuron *layer = (LayerNeuron *)layer_base;
    layer_neuron_free(layer);
}

LayerNeuron *layer_neuron_create(int size, int input_size, Activator *activator, float learning_rate) {
    LayerNeuron *layer = (LayerNeuron *)malloc(sizeof(LayerNeuron));
    layer->neurons = (Neuron **)malloc(size * sizeof(Neuron *));
    layer->neurons_size = size;
    layer->input_size = input_size;
    layer->learning_rate = learning_rate;

    activator = activator ? activator : activator_equal();
    layer->activator = activator ? activator : activator_equal();
    for (int i = 0; i < size; i++) {
        layer->neurons[i] = neuron_create(input_size, activator);
    }

    layer->layer.forward = _layer_neuron_forward;
    layer->layer.backward = _layer_neuron_backward;
    layer->layer.free = _layer_neuron_free;
    layer->input = 0;
    layer->output = 0;
    return layer;
}

LayerNeuron *layer_neuron_create_layze(int size, Activator *activator, float learning_rate) {
    LayerNeuron *layer = (LayerNeuron *)malloc(sizeof(LayerNeuron));
    layer->neurons = (Neuron **)malloc(size * sizeof(Neuron *));
    layer->neurons_size = size;
    layer->input_size = 0;
    layer->learning_rate = learning_rate;
    layer->activator = activator ? activator : activator_equal();

    for (int i = 0; i < size; i++) {
        layer->neurons[i] = 0;
    }

    layer->layer.forward = _layer_neuron_forward;
    layer->layer.backward = _layer_neuron_backward;
    layer->layer.free = _layer_neuron_free;
    layer->input = 0;
    layer->output = 0;
    layer->layze = 1;
    return layer;
}

void layer_neuron_free(LayerNeuron *layer) {
    for (int i = 0; i < layer->neurons_size; i++) {
        neuron_free(layer->neurons[i]);
    }
    free(layer->neurons);
    vector_free(layer->input);
    vector_free(layer->output);
    free(layer);
}
