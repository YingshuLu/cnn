#include <assert.h>
#include <math.h>
#include "loss.h"

LossFunction* _binary_cross_entropy_loss_instance = NULL;
LossFunction* _mean_squared_error_loss_instance = NULL;

float _binary_cross_entropy(Vector *predicted, Vector *target) {
    assert(predicted->size == target->size);
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
Vector* _binary_cross_entropy_gradient(Vector *predicted, Vector *target) {
    assert(predicted->size == target->size);
    float epsilon = 1e-15;
    Vector *gradient = vector_create(predicted->size);
    
    for (int i = 0; i < predicted->size; i++) {
        float pred = fmax(epsilon, fmin(1.0f - epsilon, predicted->data[i]));
        // BCE梯度: (p-y)/(p*(1-p))
        gradient->data[i] = (pred - target->data[i]) / (pred * (1.0f - pred));
    }
    
    return gradient;
}

float _mean_squared_error(Vector *predicted, Vector *target) {
    assert(predicted->size == target->size);
    float loss = 0.0f;
    for (int i = 0; i < predicted->size; i++) {
        loss += pow(predicted->data[i] - target->data[i], 2);
    }
    return loss / predicted->size;
}

Vector* _mean_squared_error_gradient(Vector *predicted, Vector *target) {
    assert(predicted->size == target->size);
    Vector *gradient = vector_create(predicted->size);
    for (int i = 0; i < predicted->size; i++) {
        gradient->data[i] = predicted->data[i] - target->data[i];
    }
    return gradient;
}

LossFunction* _loss_function_create(float (*loss)(Vector *predicted, Vector *target), Vector* (*gradient)(Vector *predicted, Vector *target)) {
    LossFunction* loss_function = (LossFunction*)malloc(sizeof(LossFunction));
    loss_function->loss = loss;
    loss_function->gradient = gradient;
    return loss_function;
}

LossFunction* loss_binary_cross_entropy() {
    if (_binary_cross_entropy_loss_instance == NULL) {
        _binary_cross_entropy_loss_instance = _loss_function_create(_binary_cross_entropy, _binary_cross_entropy_gradient);
    }
    return _binary_cross_entropy_loss_instance;
}

LossFunction* loss_mean_squared_error() {
    if (_mean_squared_error_loss_instance == NULL) {
        _mean_squared_error_loss_instance = _loss_function_create(_mean_squared_error, _mean_squared_error_gradient);
    }
    return _mean_squared_error_loss_instance;
}
