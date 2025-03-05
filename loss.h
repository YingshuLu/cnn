#ifndef LOSS_H
#define LOSS_H

#include "vector.h"

typedef struct LossFunction {
    float (*loss)(Vector *predicted, Vector *target);
    Vector* (*gradient)(Vector *predicted, Vector *target);
} LossFunction;

LossFunction* loss_binary_cross_entropy();
LossFunction* loss_mean_squared_error();

#endif
