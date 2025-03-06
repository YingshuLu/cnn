#include <assert.h>
#include "conv2d_layer.h"

Tensor *_conv2d_layer_forward(void *layer, Tensor *input) {
    return conv2d_layer_forward((Conv2DLayer *)layer, input);
}

Tensor *_conv2d_layer_backward(void *layer, Tensor *gradient) {
    return conv2d_layer_backward((Conv2DLayer *)layer, gradient);
}

void _conv2d_layer_free(void *layer) {
    conv2d_layer_free((Conv2DLayer *)layer);
}

Conv2DLayer *conv2d_layer_create(int in_channels, int out_channels, int kernel_size, int stride, int padding, float learning_rate, Activator* activator) {
    Conv2DLayer *layer = malloc(sizeof(Conv2DLayer));
    layer->in_channels = in_channels;
    layer->out_channels = out_channels;
    layer->kernels = calloc(out_channels, sizeof(ConvKernel *));
    layer->kernel_size = kernel_size;
    layer->learning_rate = learning_rate;
    for (int i = 0; i < out_channels; i++) {
        layer->kernels[i] = conv_kernel_create(tensor_create(kernel_size, kernel_size, in_channels), 0.0f, learning_rate, stride, padding, activator);
    }
    layer->input = 0;
    layer->layer.forward = _conv2d_layer_forward;
    layer->layer.backward = _conv2d_layer_backward;
    layer->layer.free = _conv2d_layer_free;
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
    assert(bias->size == layer->out_channels);
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

Tensor *conv2d_layer_backward(Conv2DLayer *layer, Tensor *gradient) {
    Tensor **input_gradients = (Tensor **)calloc(layer->out_channels, sizeof(Tensor *));
    Tensor *input = layer->input;

    #pragma omp parallel for
    for (int i = 0; i < layer->out_channels; i++) {
        Tensor *single_grandient = tensor_slice_refer(gradient, i, i + 1);
        input_gradients[i] = conv_kernel_backward(layer->kernels[i], input, single_grandient);
        tensor_unrefer(single_grandient);
    }

    Tensor *input_gradient = input_gradients[0];
    for (int i = 1; i < layer->out_channels; i++) {
        tensor_add(input_gradient, input_gradients[i]);
        tensor_unrefer(input_gradients[i]);
    }

    free(input_gradients);
    return input_gradient;
}