#include <assert.h>
#include "conv2d_layer.h"

Conv2DLayer *conv2d_layer_create(int in_channels, int out_channels, int kernel_size, int stride, int padding, Activation* activation) {
    Conv2DLayer *layer = malloc(sizeof(Conv2DLayer));
    layer->in_channels = in_channels;
    layer->out_channels = out_channels;
    layer->kernels = calloc(out_channels, sizeof(ConvKernel *));
    layer->kernel_size = kernel_size;
    for (int i = 0; i < out_channels; i++) {
        layer->kernels[i] = conv_kernel_create(tensor_create(kernel_size, kernel_size, in_channels), 0.0f, stride, padding, activation);
    }
    layer->input = 0;
    return layer;
}

void conv2d_layer_free(Conv2DLayer *layer) {
    for (int i = 0; i < layer->out_channels; i++) {
        conv_kernel_free(layer->kernels[i]);
    }
    free(layer->kernels);
    free(layer);
}

void conv2d_layer_init_bias(Conv2DLayer *layer, Vector *bias) {
    for (int i = 0; i < layer->out_channels; i++) {
        layer->kernels[i]->bias = bias->data[i];
    }
}

Tensor *conv2d_layer_forward(Conv2DLayer *layer, Tensor *input) {
    Tensor **kernel_outputs = (Tensor **)calloc(layer->out_channels, sizeof(Tensor *));

    tensor_unrefer(layer->input);
    layer->input = tensor_refer(input);

    #pragma omp parallel for
    for (int i = 0; i < layer->out_channels; i++) {
        kernel_outputs[i] = conv_kernel_forward(layer->kernels[i], input);
    }

    Tensor *output = kernel_outputs[0];
    for (int i = 1; i < layer->out_channels; i++) {
        tensor_concat_refer(output, kernel_outputs[i]);
        tensor_unrefer(kernel_outputs[i]);
    }

    free(kernel_outputs);
    return output;
}
