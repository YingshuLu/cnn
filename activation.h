#ifndef ACTIVATION_H
#define ACTIVATION_H

typedef struct Activation {
    float (*activate)(float);
    float (*derivate)(float);
} Activation;

Activation *activation_sigmoid();
Activation *activation_relu();
Activation *activation_leaky_relu();

#endif