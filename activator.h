#ifndef ACTIVATOR_H
#define ACTIVATOR_H

typedef struct Activator {
    float (*activate)(float);
    float (*derivate)(float);
} Activator;

Activator *activator_sigmoid();
Activator *activator_relu();
Activator *activator_leaky_relu();
Activator *activator_equal();

#endif