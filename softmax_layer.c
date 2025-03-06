#include <math.h>
#include "softmax_layer.h"

Vector *_vector_softmax(Vector *input){
    Vector *output = vector_create(input->size);
    float sum = 0.0f;
    for (int i = 0; i < input->size; i++) {
        sum += exp(input->data[i]);
    }
    if (sum == 0.0f) {
        return 0;
    }

    for (int i = 0; i < input->size; i++) {
        output->data[i] = exp(input->data[i]) / sum;
    }
    return output;
}

Vector *_vector_softmax_derivative(Vector *input, Vector *gradient){
    Vector *output = vector_create(input->size);
    for (int i = 0; i < input->size; i++) {
        output->data[i] = gradient->data[i] * input->data[i] * (1.0f - input->data[i]);
    }
    return output;
}

Vector *_softmax_layer_forward(void *layer, Vector *input) {
    SoftmaxLayer *softmax_layer = (SoftmaxLayer *)layer;
    softmax_layer->input = vector_refer(input);
    return _vector_softmax(softmax_layer->input);
}

Vector *_softmax_layer_backward(void *layer, Vector *gradient) {
    SoftmaxLayer *softmax_layer = (SoftmaxLayer *)layer;
    return _vector_softmax_derivative(softmax_layer->input, gradient);
}

void _softmax_layer_free(void *layer) {
    SoftmaxLayer *softmax_layer = (SoftmaxLayer *)layer;
    free(softmax_layer);
}

SoftmaxLayer *softmax_layer_create() {
    SoftmaxLayer *layer = (SoftmaxLayer *)malloc(sizeof(SoftmaxLayer));
    layer->input = 0;
    layer->layer.forward = _softmax_layer_forward;
    layer->layer.backward = _softmax_layer_backward;
    layer->layer.free = _softmax_layer_free;
    return layer;
}

void softmax_layer_free(SoftmaxLayer *layer) {
    vector_unrefer(layer->input);
    free(layer);
}

Vector *softmax_layer_forward(SoftmaxLayer *layer, Vector *input) {
    vector_unrefer(layer->input);
    layer->input = vector_refer(input);
    return _vector_softmax(layer->input);
}

Vector *softmax_layer_backward(SoftmaxLayer *layer, Vector *gradient) {
    Vector *new_gradient = _vector_softmax_derivative(layer->input, gradient);
    vector_unrefer(gradient);
    return new_gradient;
}