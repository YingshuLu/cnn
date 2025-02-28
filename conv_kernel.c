#include <assert.h>
#include "conv_kernel.h"

Tensor *_tensor_conv2d(Tensor *input, Tensor *kernel, int stride, int padding, float bias) {
    assert(kernel->depth == input->depth &&
            kernel->rows <= input->rows &&
            kernel->cols <= input->cols);

    int output_rows = (input->rows - kernel->rows + 2 * padding) / stride + 1;
    int output_cols = (input->cols - kernel->cols + 2 * padding) / stride + 1;
    Tensor *output = tensor_create(output_rows, output_cols, 1);

    #pragma omp parallel for collapse(2)
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

ConvKernel *conv_kernel_create(Tensor *tensor, float bias, int stride, int padding, Activation *activation) {
    ConvKernel *kernel = malloc(sizeof(ConvKernel));
    kernel->tensor = tensor;
    kernel->bias = bias;
    kernel->stride = stride;
    kernel->padding = padding;
    kernel->activation = activation;
    kernel->linear_output = 0;
    return kernel;
}

void conv_kernel_free(ConvKernel *kernel) {
    tensor_free(kernel->tensor);
    free(kernel);
}

Tensor *conv_kernel_forward(ConvKernel *kernel, Tensor *input) {
    tensor_unrefer(kernel->linear_output);
    Tensor *linear_output = _tensor_conv2d(input, kernel->tensor, kernel->stride, kernel->padding, kernel->bias);
    kernel->linear_output = tensor_refer(linear_output);

    Tensor *output = linear_output;
    if (kernel->activation) {
        Tensor *output = tensor_copy(linear_output);
        tensor_apply(output, kernel->activation->activate);
    }
    return output;
}