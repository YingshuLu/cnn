#include <stdio.h>
#include "network.h"
#include "neuron_layer.h"

float binary_predict(float v) {
    return v > 0.5 ? 1.0 : 0.0;
}

int main() {

    int input_size = 8;
    int output_size = 3;
    int samples_count = 8;

    float train_data[][8] = {
        {0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0},
        {1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0},
        {0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0},
        {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0},
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0},
        {0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0},
        };

    float label_data[][3] = {
        {0.0f, 1.0f, 0.0f},
        {0.0f, 1.0f, 1.0f},
        {0.0f, 1.0f, 1.0f}, 
        {1.0f, 0.0f, 0.0f},
        {1.0f, 1.0f, 1.0f},
        {0.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
        {1.0f, 1.0f, 0.0f},
    };

    float test_input_data[] = {0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0};

    Vector **inputs = (Vector **)calloc(samples_count, sizeof(Vector *));
    Vector **labels = (Vector **)calloc(samples_count, sizeof(Vector *));

    for (int i = 0; i < samples_count; i++) {
        inputs[i] = vector_create_from_array(train_data[i], input_size);
        labels[i] = vector_create_from_array(label_data[i], output_size);
    }

    Network *network = network_create();
    LayerNeuron *layer0 = layer_neuron_create(8, input_size, activation_relu(), 0.001);
    network_add_layer(network, (Layer*)layer0);
    LayerNeuron *layer1 = layer_neuron_create(3, layer0->neurons_size, activation_sigmoid(), 0.001);
    network_add_layer(network, (Layer*)layer1);

    network_train(network, inputs, samples_count, labels, 10000);

    Vector *test_input = vector_create_from_array(test_input_data, input_size);
    Vector *output = network_predict(network, test_input);
    printf("Test Input: [%f, %f, %f, %f, %f, %f, %f, %f] RawOutput: [%f, %f, %f] BinaryOutput: [%f, %f, %f]\n", 
        test_input->data[0], test_input->data[1], test_input->data[2], test_input->data[3], test_input->data[4], test_input->data[5], test_input->data[6], test_input->data[7],
        output->data[0], output->data[1], output->data[2],
        binary_predict(output->data[0]), binary_predict(output->data[1]), binary_predict(output->data[2]));

    for (int i = 0; i < input_size; i++) {
        output = network_predict(network, inputs[i]);
        printf("Input: [%f, %f, %f, %f, %f, %f, %f, %f] RawOutput: [%f, %f, %f] BinaryOutput: [%f, %f, %f]\n", 
            inputs[i]->data[0], inputs[i]->data[1], inputs[i]->data[2], inputs[i]->data[3], inputs[i]->data[4], inputs[i]->data[5], inputs[i]->data[6], inputs[i]->data[7],
            output->data[0], output->data[1], output->data[2],
            binary_predict(output->data[0]), binary_predict(output->data[1]), binary_predict(output->data[2]));
    }
}