#include <math.h>
#include <stdlib.h>
#include "activation.h"

Activation *sigmoid_instance = 0;
Activation *relu_instance = 0;
Activation *leaky_relu_instance = 0;
Activation *equal_instance = 0;

float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

float relu(float x) {
    return x > 0 ? x : 0.0f;
}

float sigmoid_derivative(float x) {
    return sigmoid(x) * (1.0 - sigmoid(x));
}

float relu_derivative(float x) {
    return x > 0 ? 1.0f : 0.0f;
}

float leaky_relu(float x) {
    return x > 0 ? x : 0.01f * x;
}

float leaky_relu_derivative(float x) {
    return x > 0 ? 1.0f : 0.01f;
}

float equal(float x) {
    return x;
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
float equal_derivative(float x) {
    return 1.0f;
}
#pragma GCC diagnostic pop

Activation *activation_create() {
    return (Activation *)malloc(sizeof(Activation));
}

Activation *activation_sigmoid() {
    if (!sigmoid_instance) {
        sigmoid_instance = activation_create();
        sigmoid_instance->activate = sigmoid;
        sigmoid_instance->derivate = sigmoid_derivative;
    }
    return sigmoid_instance;
}

Activation *activation_relu() {
    if (!relu_instance) {
        relu_instance = activation_create();
        relu_instance->activate = relu;
        relu_instance->derivate = relu_derivative;
    }
    return relu_instance;
}

Activation *activation_leaky_relu() {
    if (!leaky_relu_instance) {
        leaky_relu_instance = activation_create();
        leaky_relu_instance->activate = leaky_relu;
        leaky_relu_instance->derivate = leaky_relu_derivative;
    }
    return leaky_relu_instance;
}

Activation *activation_equal() {
    if (!equal_instance) {
        equal_instance = activation_create();
        equal_instance->activate = equal;
        equal_instance->derivate = equal_derivative;
    }
    return equal_instance;
}
