#ifndef CONV3D_LAYER_H
#define CONV3D_LAYER_H

#include "tensor.h"
#include "conv_kernel.h"

typedef struct Conv2DLayer {
    ConvKernel **kernels;
    int kernel_size;
    int in_channels;
    int out_channels;
    Tensor *input;
} Conv2DLayer;

Conv2DLayer *conv2d_layer_create(int in_channels, int out_channels, int kernel_size, int stride, int padding, Activation* activation);
void conv2d_layer_init_bias(Conv2DLayer *layer, Vector *bias);
void conv2d_layer_free(Conv2DLayer *layer);
Tensor *conv2d_layer_forward(Conv2DLayer *layer, Tensor *input);
Tensor *conv2d_layer_backward(Conv2DLayer *layer, Tensor *input, Tensor *output, Tensor *d_output);

#endif