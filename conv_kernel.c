#include <assert.h>
#include "conv_kernel.h"

Tensor *_tensor_conv2d(Tensor *input, Tensor *kernel, int stride, int padding, float bias) {
    assert(kernel->depth == input->depth &&
            kernel->rows <= input->rows &&
            kernel->cols <= input->cols);

    int output_rows = (input->rows - kernel->rows + 2 * padding) / stride + 1;
    int output_cols = (input->cols - kernel->cols + 2 * padding) / stride + 1;
    Tensor *output = tensor_create(output_rows, output_cols, 1);

    for (int i = 0; i < output_rows; i++) {
        for (int j = 0; j < output_cols; j++) {
            float sum = 0.0f;
            for (int d = 0; d < input->depth; d++) {
                for (int k = 0; k < kernel->rows; k++) {
                    for (int l = 0; l < kernel->cols; l++) {
                        int ix = i * stride + k - padding;
                        int iy = j * stride + l - padding;
                        if (ix >= 0 && ix < input->rows && iy >= 0 && iy < input->cols) {
                            sum += tensor_get(input, ix, iy, d) * tensor_get(kernel, k, l, d);
                        }
                    }
                }
            }
            tensor_set(output, i, j, 0, sum + bias);
        }
    }
    return output;
}

ConvKernel *conv_kernel_create(Tensor *tensor, float bias, float learning_rate, int stride, int padding, Activation *activation) {
    ConvKernel *kernel = malloc(sizeof(ConvKernel));
    kernel->tensor = tensor;
    kernel->bias = bias;
    kernel->learning_rate = learning_rate;
    kernel->stride = stride;
    kernel->padding = padding;
    kernel->activation = activation;
    kernel->conv_output = 0;
    return kernel;
}

void conv_kernel_free(ConvKernel *kernel) {
    tensor_free(kernel->tensor);
    free(kernel);
}

Tensor *conv_kernel_forward(ConvKernel *kernel, Tensor *input) {
    tensor_unrefer(kernel->conv_output);
    Tensor *conv_output = _tensor_conv2d(input, kernel->tensor, kernel->stride, kernel->padding, kernel->bias);
    kernel->conv_output = tensor_refer(conv_output);

    Tensor *output = conv_output;
    if (kernel->activation) {
        Tensor *output = tensor_copy(conv_output);
        tensor_apply(output, kernel->activation->activate);
    }
    return output;
}

Tensor *conv_kernel_backward(ConvKernel *kernel, Tensor *input, Tensor *gradient) {
    // Initialize gradients for input and kernel
    Tensor *input_gradient = tensor_create(input->rows, input->cols, input->depth);
    Tensor *kernel_gradient = tensor_create(kernel->tensor->rows, kernel->tensor->cols, kernel->tensor->depth);
    float bias_gradient = tensor_sum(gradient);

    // Compute gradients
    for (int i = 0; i < input->rows; i++) {
        for (int j = 0; j < input->cols; j++) {
            for (int d = 0; d < input->depth; d++) {
                for (int k = 0; k < kernel->tensor->rows; k++) {
                    for (int l = 0; l < kernel->tensor->cols; l++) {
                        int ix = i + k - kernel->padding;
                        int iy = j + l - kernel->padding;
                        if (ix >= 0 && ix < input->rows && iy >= 0 && iy < input->cols) {
                            float grad_value = tensor_get(gradient, i, j, 0);

                            float input_value = tensor_get(input_gradient, ix, iy, d) + grad_value * tensor_get(kernel->tensor, k, l, d);
                            tensor_set(input_gradient, ix, iy, d, input_value);

                            float kernel_value = tensor_get(kernel_gradient, k, l, d) + grad_value * tensor_get(input, ix, iy, d);
                            tensor_set(kernel_gradient, k, l, d, kernel_value);
                        }
                    }
                }
            }
        }
    }

    // Update kernel weights and return input gradient
    tensor_mul_value(kernel_gradient, kernel->learning_rate);
    tensor_sub(kernel->tensor, kernel_gradient);
    tensor_free(kernel_gradient);

    kernel->bias -= kernel->learning_rate * bias_gradient;
    return input_gradient;
}
