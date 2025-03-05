#ifndef CNN_H
#define CNN_H

#include "tensor.h"
#include "vector.h"
#include "network.h"
#include "tensor_network.h"
typedef struct CNN {
    TensorNetwork *tensor_network;
    Network *network;
} CNN;

CNN *cnn_create(TensorNetwork *tensor_network, Network *network);
void cnn_free(CNN *cnn);
Vector *cnn_train(CNN *cnn, Tensor **samples, int samples_count, Vector **lables, int epochs);
Vector *cnn_predict(CNN *cnn, Tensor *input);

#endif
