#include "max_pool_layer.h"

Tensor *_max_pool_layer_forward(void *layer, Tensor *input) {
    return max_pool_layer_forward((MaxPoolLayer *)layer, input);
}

Tensor *_max_pool_layer_backward(void *layer, Tensor *gradient) {
    return max_pool_layer_backward((MaxPoolLayer *)layer, gradient);
}

void _max_pool_layer_free(void *layer) {
    max_pool_layer_free((MaxPoolLayer *)layer);
}

MaxPoolLayer *max_pool_layer_create(int pool_size, int stride) {
    MaxPoolLayer *layer = (MaxPoolLayer *)malloc(sizeof(MaxPoolLayer));
    layer->pool_size = pool_size;
    layer->stride = stride;
    layer->output_mask = 0;
    layer->layer.forward = _max_pool_layer_forward;
    layer->layer.backward = _max_pool_layer_backward;
    layer->layer.free = _max_pool_layer_free;
    return layer;
}

void max_pool_layer_free(MaxPoolLayer *layer) {
    if (layer->output_mask) {
        tensor_unrefer(layer->output_mask);
    }
    free(layer);
}

Tensor *max_pool_layer_forward(MaxPoolLayer *layer, Tensor *input) {
    int output_rows = (input->rows - layer->pool_size) / layer->stride + 1;
    int output_cols = (input->cols - layer->pool_size) / layer->stride + 1;

    Tensor *output = tensor_create(output_rows, output_cols, input->depth);

    if (!layer->output_mask) {
        layer->output_mask = tensor_create(output->rows, output->cols, 2 * output->depth);
    } else {
        tensor_fill_value(layer->output_mask, 0.0f);
    }

    for (int d = 0; d < output->depth; d++) {
        for (int i = 0; i < output->rows; i++) {
            for (int j = 0; j < output->cols; j++) {
                int ix = i * layer->stride;
                int iy = j * layer->stride;
                float max_value = tensor_get(input, ix, iy, d);
                int max_ix = ix;
                int max_iy = iy;
                for (int k = 0; k < layer->pool_size; k++) {
                    for (int l = 0; l < layer->pool_size; l++) {
                        float value = tensor_get(input, ix + k, iy + l, d);
                        if (value > max_value) {
                            max_value = value;
                            max_ix = ix + k;
                            max_iy = iy + l;
                        }
                    }
                }
                tensor_set(output, i, j, d, max_value);
                tensor_set(layer->output_mask, i, j, 2*d, max_ix);
                tensor_set(layer->output_mask, i, j, 2*d+1, max_iy);
            }
        }
    }   
    return output;
}

Tensor *max_pool_layer_backward(MaxPoolLayer *layer, Tensor *gradient) {
    Tensor *input_gradient = tensor_create(layer->output_mask->rows * layer->stride, 
        layer->output_mask->cols * layer->stride, 
        gradient->depth);

    for (int d = 0; d < gradient->depth; d++) {
        for (int i = 0; i < gradient->rows; i++) {
            for (int j = 0; j < gradient->cols; j++) {
                int max_x = (int)tensor_get(layer->output_mask, i, j, 2*d);
                int max_y = (int)tensor_get(layer->output_mask, i, j, 2*d+1);
                float current = tensor_get(input_gradient, max_x, max_y, d);
                tensor_set(input_gradient, max_x, max_y, d, current + tensor_get(gradient, i, j, d));
            }
        }
    }
    return input_gradient;
}

