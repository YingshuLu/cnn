#ifndef ACTIVATION_H
#define ACTIVATION_H

typedef struct Activator {
    float (*activate)(float);
    float (*derivate)(float);
} Activator;

Activator *activation_sigmoid();
Activator *activation_relu();
Activator *activation_leaky_relu();
Activator *activation_equal();

#endif