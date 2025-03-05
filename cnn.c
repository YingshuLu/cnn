#include "cnn.h"
#include "loss.h"
#include "shuffle.h"

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
    Tensor *output = 0, *tensor_gradient = 0;
    Vector *predicted = 0, *flatten = 0, *gradient = 0;
    LossFunction *loss_function = loss_binary_cross_entropy();
    Shuffle *shuffle = shuffle_create(samples_count);
    int index = 0;
    for (int epoch = 0; epoch < epochs; epoch++) {
        float epoch_loss = 0.0;
        shuffle_shuffle(shuffle);
        while (!shuffle_end(shuffle)) {
            index = shuffle_next(shuffle);
            output = tensor_network_forward(cnn->tensor_network, samples[index]);
            flatten = tensor_flatten(output);
            predicted = network_forward(cnn->network, flatten);

            epoch_loss += loss_function->loss(predicted, lables[index]);
            gradient = loss_function->gradient(predicted, lables[index]);
            gradient = network_backward(cnn->network, gradient);
            tensor_gradient = vector_to_tensor(gradient, output->rows, output->cols, output->depth);
            tensor_gradient = tensor_network_backward(cnn->tensor_network, tensor_gradient);
            
            tensor_unrefer(output);
            tensor_unrefer(tensor_gradient);

            vector_unrefer(predicted);
            vector_unrefer(flatten);
            vector_unrefer(gradient);
        }
        if (epoch_loss < 0.15) {
            break;
        }
    }
    shuffle_free(shuffle);
    return predicted;
}

Vector *cnn_predict(CNN *cnn, Tensor *input) {
    Tensor *output = tensor_network_forward(cnn->tensor_network, input);
    Vector *flatten = tensor_flatten(output);
    Vector *predicted = network_forward(cnn->network, flatten);
    tensor_free(output);
    vector_free(flatten);
    return predicted;
}