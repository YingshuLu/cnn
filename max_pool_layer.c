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
    layer->input_mask = 0;
    layer->layer.forward = _max_pool_layer_forward;
    layer->layer.backward = _max_pool_layer_backward;
    layer->layer.free = _max_pool_layer_free;
    return layer;
}

void max_pool_layer_free(MaxPoolLayer *layer) {
    if (layer->input_mask) {
        tensor_unrefer(layer->input_mask);
    }
    free(layer);
}

Tensor *max_pool_layer_forward(MaxPoolLayer *layer, Tensor *input) {
    if (!layer->input_mask) {
        layer->input_mask = tensor_create(input->rows, input->cols, input->depth);
    } else {
        tensor_fill_value(layer->input_mask, 0.0f);
    }

    int output_rows = (input->rows - layer->pool_size) / layer->stride + 1;
    int output_cols = (input->cols - layer->pool_size) / layer->stride + 1;

    Tensor *output = tensor_create(output_rows, output_cols, input->depth);
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
                tensor_set(layer->input_mask, max_ix, max_iy, d, 1.0f);
            }
        }
    }   
    return output;
}

Tensor *max_pool_layer_backward(MaxPoolLayer *layer, Tensor *gradient) {
    Tensor *input_gradient = tensor_create(layer->input_mask->rows, layer->input_mask->cols, layer->input_mask->depth);
    for (int d = 0; d < gradient->depth; d++) {
        for (int i = 0; i < gradient->rows; i++) {
            for (int j = 0; j < gradient->cols; j++) {
                int ix = i * layer->stride;
                int iy = j * layer->stride;
                for (int k = 0; k < layer->pool_size; k++) {
                    for (int l = 0; l < layer->pool_size; l++) {
                        float mask_value = tensor_get(layer->input_mask, ix + k, iy + l, d);
                        if (mask_value > 0.0f) {
                            float value = mask_value *tensor_get(gradient, i, j, d) +
                                    tensor_get(input_gradient, ix + k, iy + l, d);
                            tensor_set(input_gradient, ix + k, iy + l, d, value);
                        }
                    }
                }
            }
        }
    }
    return input_gradient;
}

