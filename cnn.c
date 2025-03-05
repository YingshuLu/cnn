#include "cnn.h"

CNN *cnn_create(TensorNetwork *tensor_network, Network *network) {
    CNN *cnn = (CNN *)malloc(sizeof(CNN));
    cnn->tensor_network = tensor_network;
    cnn->network = network;
    return cnn;
}

void cnn_free(CNN *cnn) {
    tensor_network_free(cnn->tensor_network);
    network_free(cnn->network);
    free(cnn);
}   

Vector *cnn_train(CNN *cnn, Tensor **samples, int samples_count, Vector **lables, int epochs) {
    return 0;
}

Vector *cnn_predict(CNN *cnn, Tensor *input) {
    return 0;
}