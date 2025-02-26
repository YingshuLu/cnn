#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include "network.h"
#include "neuron_layer.h"

float binary_cross_entropy(Vector *predicted, Vector *target) {
    float loss = 0.0f;
    float epsilon = 1e-15; // 防止log(0)
    
    for (int i = 0; i < predicted->size; i++) {
        // 限制预测值在(0,1)区间
        float pred = fmax(epsilon, fmin(1.0f - epsilon, predicted->data[i]));
        // 二元交叉熵公式: -[y*log(p) + (1-y)*log(1-p)]
        loss -= target->data[i] * log(pred) + (1.0f - target->data[i]) * log(1.0f - pred);
    }
    
    return loss / predicted->size; // 归一化
}

// 交叉熵的梯度计算
Vector* binary_cross_entropy_gradient(Vector *predicted, Vector *target) {
    float epsilon = 1e-15;
    Vector *gradient = vector_create(predicted->size);
    
    for (int i = 0; i < predicted->size; i++) {
        float pred = fmax(epsilon, fmin(1.0f - epsilon, predicted->data[i]));
        // BCE梯度: (p-y)/(p*(1-p))
        gradient->data[i] = (pred - target->data[i]) / (pred * (1.0f - pred));
    }
    
    return gradient;
}

Network *network_create() {
    Network *network = (Network *)malloc(sizeof(Network));
    network->capacity = 4;
    network->layers = (Layer **)calloc(network->capacity, sizeof(Layer *));
    network->count = 0;
    return network;
}

void network_add_layer(Network *network, Layer *layer) {
    if (network->count >= network->capacity) {
        network->capacity *= 2;
        network->layers = (Layer **)realloc(network->layers, network->capacity * sizeof(Layer *));
    }
    network->layers[network->count] = layer;
    network->count++;
}

Vector *network_predict(Network *network, Vector *input) {
    Vector *output;
    Layer *layer;
    for (int i = 0; i < network->count; i++) {
        layer = network->layers[i];
        output = layer->forward(layer, input);
        input = output;
    }
    return output;
}

void network_train(Network *network, Vector **samples, int samples_count, Vector **labels, int epochs) {
    Vector *predicted, *gradient;
    for (int epoch = 0; epoch < epochs; epoch++) {
        float epoch_loss = 0.0;
        for (int i = 0; i < samples_count; i++) {
            predicted = network_predict(network, samples[i]);
            
            /*
            //print predicted and label
            //printf("--------------------------------\n");
            //printf("Predicted: [%f, %f, %f]\n", predicted->data[0], predicted->data[1], predicted->data[2]);
            //printf("Label: [%f, %f, %f]\n", label->data[0], label->data[1], label->data[2]);
            vector_sub(predicted, labels[i]);
            float loss_pow = vector_dot(predicted, predicted);
            epoch_loss += sqrt(loss_pow);
            printf("- Epoch %d sample_loss: %f\n", epoch, sqrt(loss_pow));
            //printf("--------------------------------\n");

            gradient = predicted;
            */

           // 计算交叉熵损失
            float loss = binary_cross_entropy(predicted, labels[i]);
            epoch_loss += loss;
            
            // 计算交叉熵梯度
            gradient = binary_cross_entropy_gradient(predicted, labels[i]);
            
            for (int j = network->count - 1; j >= 0; j--) {
                gradient = network->layers[j]->backward(network->layers[j], gradient);
            }
            vector_destroy(predicted);
        }

        epoch_loss /= samples_count;
        printf("+ Epoch %d averate_loss: %f\n", epoch, epoch_loss);
        if (epoch_loss < 0.15) {
            break;
        }
    }
}

void network_destroy(Network *network) {
    for (int i = 0; i < network->count; i++) {
        free(network->layers[i]);
    }
    free(network->layers);
    free(network);
}