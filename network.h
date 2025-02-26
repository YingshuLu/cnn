#ifndef NETWORK_H
#define NETWORK_H

#include "layer.h"

typedef struct Network {
    Layer **layers;
    int count;
    int capacity;
} Network;

Network *network_create();

void network_add_layer(Network *network, Layer *layer);

Vector *network_predict(Network *network, Vector *input);

void network_train(Network *network, Vector **samples, int samples_count, Vector **labels, int epochs);

void network_destroy(Network *network);

#endif
