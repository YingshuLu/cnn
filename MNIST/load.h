#ifndef MNIST_LOAD_H
#define MNIST_LOAD_H

#include "tensor.h"

typedef struct MNISTData {
    Tensor **images;
    Vector **labels;
    int count;
} MNISTData;

MNISTData *load_mnist_data(const char *images_path, const char *labels_path);

void free_mnist_data(MNISTData *data);

#endif
