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
            for (int d = 0; d < kernel->depth; d++) {
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
    tensor_unrefer(kernel->conv_output);
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

/*
       [Example]
       Input: [M, N, 3]
       Kernel: [3, 3, 3]
       
       reshape:
       Sampling: Matrix[MN, 27]
       Kernel: Matrix[27, 1]

       forward:
       F = [MN x 27] x [27 x 1] = [MN x 1]
       backward:
       dF/Dw = [1 x MN] x [MN x 27] = [1 x 27]
       dF/Db = Sum([MN x 1]) = [1 x 1]
       dF/Dx = [MN x 1] x [1 x 27] = [MN x 27]
*/
Tensor *conv_kernel_backward(ConvKernel *kernel, Tensor *input, Tensor *gradient) {
    assert(input->depth == kernel->tensor->depth &&
           gradient->rows == kernel->conv_output->rows &&
           gradient->cols == kernel->conv_output->cols);

    Tensor *input_gradient = tensor_create(input->rows, input->cols, input->depth);
    float bias_gradient = 0.0f;

    for (int i = 0; i < gradient->rows; i++) {
        for (int j = 0; j < gradient->cols; j++) {
            float grad_value = tensor_get(gradient, i, j, 0) * 
                        kernel->activation->derivate(tensor_get(kernel->conv_output, i, j, 0));
            bias_gradient += grad_value;

            for (int d = 0; d < kernel->tensor->depth; d++) {
              for (int k = 0; k < kernel->tensor->rows; k++) {
                for (int l = 0; l < kernel->tensor->cols; l++) {
                    int ix = i * kernel->stride + k - kernel->padding;
                    int iy = j * kernel->stride + l - kernel->padding;
                    if (ix >= 0 && ix < input->rows && iy >= 0 && iy < input->cols) {
                        // 输入梯度：使用翻转的卷积核
                        float input_value = tensor_get(input_gradient, ix, iy, d) + 
                                      grad_value * tensor_get(kernel->tensor, 
                                                                 kernel->tensor->rows-1-k, 
                                                                 kernel->tensor->cols-1-l, d);
                        tensor_set(input_gradient, ix, iy, d, input_value);

                        // 卷积核梯度：保持不变
                        float kernel_value = tensor_get(kernel->tensor, k, l, d) - 
                                        kernel->learning_rate * grad_value * tensor_get(input, ix, iy, d);
                        tensor_set(kernel->tensor, k, l, d, kernel_value);
                        }
                    }
                }
            }
        }
    }

    kernel->bias -= kernel->learning_rate * bias_gradient;
    return input_gradient;
}
