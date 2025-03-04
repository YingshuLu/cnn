#ifndef CONV_KERNEL_H
#define CONV_KERNEL_H

#include "activation.h"
#include "tensor.h"

typedef struct ConvKernel {
    Tensor *tensor;
    float bias;
    float learning_rate;
    int stride;
    int padding;
    Activator *activation;
    Tensor *conv_output;
} ConvKernel;

ConvKernel *conv_kernel_create(Tensor *kernel, float bias, float learning_rate, int stride, int padding, Activator *activation);
void conv_kernel_free(ConvKernel *kernel);
Tensor *conv_kernel_forward(ConvKernel *kernel, Tensor *input);
Tensor *conv_kernel_backward(ConvKernel *kernel, Tensor *input, Tensor *gradient);

#endif