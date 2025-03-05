#include <stdio.h>
#include "activator.h"
#include "load.h"
#include "tensor.h"
#include "network.h"
#include "conv2d_layer.h"
#include "activator.h"
#include "max_pool_layer.h"
#include "tensor_network.h"
#include "neuron_layer.h"
#include "cnn.h"

float LEARNING_RATE = 0.0001;

int max_index(Vector *vector) {
    float max = vector->data[0];
    int max_index = 0;
    for(int i = 0; i < vector->size; i++) {
        if(vector->data[i] > max) {
            max = vector->data[i];
            max_index = i;
        }
    }
    return max_index;
}

int main(int argc, char **argv) {
    MNISTData *train_data = load_mnist_data("train-images.idx3-ubyte", "train-labels.idx1-ubyte");
    MNISTData *test_data = load_mnist_data("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte");

    int label_size = train_data->labels[0]->size;

    Conv2DLayer *conv2_layer = conv2d_layer_create(1, 16, 2, 1, 0, LEARNING_RATE, activator_leaky_relu());
    MaxPoolLayer *max_pool_layer = max_pool_layer_create(2, 2);

    Conv2DLayer *conv2_layer2 = conv2d_layer_create(16, 8, 4, 1, 0, LEARNING_RATE, activator_leaky_relu());
    MaxPoolLayer *max_pool_layer2 = max_pool_layer_create(2, 2);
    
    TensorNetwork *tensor_network = tensor_network_create();
    tensor_network_add_layer(tensor_network, (TensorLayer*)conv2_layer);
    tensor_network_add_layer(tensor_network, (TensorLayer*)max_pool_layer);
    tensor_network_add_layer(tensor_network, (TensorLayer*)conv2_layer2);
    tensor_network_add_layer(tensor_network, (TensorLayer*)max_pool_layer2);

    LayerNeuron *layer0 = layer_neuron_create_layze(2 * label_size, activator_leaky_relu(), LEARNING_RATE);
    LayerNeuron *layer1 = layer_neuron_create(label_size, layer0->neurons_size, activator_sigmoid(), LEARNING_RATE);
    Network *network = network_create();
    network_add_layer(network, (Layer*)layer0);
    network_add_layer(network, (Layer*)layer1);

    CNN *cnn = cnn_create(tensor_network, network);
    cnn_train(cnn, train_data->images, train_data->count, train_data->labels, 60, 10);

    int correct = 0;
    for(int i = 0; i < test_data->count; i++) {
        Vector *predicted = cnn_predict(cnn, test_data->images[i]);
        int predicted_label = max_index(predicted);
        if (vector_get(test_data->labels[i], predicted_label) > 0.0) {
            correct++;
        }
        vector_free(predicted);
    }
    printf("correct: %d\n", correct);
    printf("accuracy: %f\n", (float)correct / test_data->count);

    return 0;
}

