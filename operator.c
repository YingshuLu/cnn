#include <math.h>
#include "operator.h"

float float_add(float a, float b) {
    return a + b;
}

float float_sub(float a, float b) {
    return a - b;
}

float float_mul(float a, float b) {
    return a * b;
}

float float_div(float a, float b) {
    return a / b;
}

float float_mod(float a, float b) {
    return fmod(a, b);
}

float float_pow(float a, float b) {
    return powf(a, b);
}

float float_fill(float a, float b) {
    return b;
}