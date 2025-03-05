#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "load.h"

const int32_t MAGIC_NUMBER_IMAGES = 2051;
const int32_t MAGIC_NUMBER_LABELS = 2049;

int32_t _net_to_host_int32(int32_t value) {
    return (value & 0xFF) << 24 |
           ((value >> 8) & 0xFF) << 16 |
           ((value >> 16) & 0xFF) << 8 |
           (value >> 24);
}

Tensor **_load_mnist_images(const char *path, int *count) {
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        printf("Error: Could not open file %s\n", path);
        return NULL;
    }

    int32_t magic_number = 0;
    fread(&magic_number, sizeof(magic_number), 1, file);
    magic_number = _net_to_host_int32(magic_number);
    if (magic_number != MAGIC_NUMBER_IMAGES) {
        printf("Error: Invalid magic number %d for images file\n", magic_number);
        fclose(file);
        return NULL;
    }

    int32_t num_images = 0;
    fread(&num_images, sizeof(num_images), 1, file);
    num_images = _net_to_host_int32(num_images);

    int32_t rows, cols;
    fread(&rows, sizeof(rows), 1, file);
    rows = _net_to_host_int32(rows);

    fread(&cols, sizeof(cols), 1, file);
    cols = _net_to_host_int32(cols);

    Tensor **images = (Tensor **)malloc(num_images * sizeof(Tensor *));
    Tensor *image;
    char pixel;
    for (int i = 0; i < num_images; i++) {
        image = tensor_create(rows, cols, 1);
        for (int k = 0; k < rows; k++) {
            for (int j = 0; j < cols; j++) {
                fread(&pixel, 1, 1, file);
                tensor_set(image, k, j, 0, (float)pixel / 255.0f);
            }
        }
        images[i] = image;
    }

    fclose(file);
    *count = num_images;
    printf("loaded from %s with %d-images\n", path, num_images);
    return images;
}

Vector **_load_mnist_labels(const char *path, int *count) {
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        printf("Error: Could not open file %s\n", path);
        return NULL;
    }

    int32_t magic_number = 0;
    fread(&magic_number, sizeof(magic_number), 1, file);
    magic_number = _net_to_host_int32(magic_number);
    if (magic_number != MAGIC_NUMBER_LABELS) {
        printf("Error: Invalid magic number %d for labels file\n", magic_number);
        fclose(file);
        return NULL;
    }

    int32_t num_labels = 0;
    fread(&num_labels, sizeof(num_labels), 1, file);
    num_labels = _net_to_host_int32(num_labels);

    Vector **labels = (Vector **)malloc(num_labels * sizeof(Vector *));
    char label;
    for (int i = 0; i < num_labels; i++) {
        fread(&label, sizeof(label), 1, file);
        labels[i] = vector_create(10);
        labels[i]->data[label] = 1.0f;
    }

    fclose(file);
    *count = num_labels;
    printf("loaded from %s with %d-labels\n", path, num_labels);
    return labels;
}

MNISTData *load_mnist_data(const char *images_path, const char *labels_path) {
    MNISTData *data = (MNISTData *)malloc(sizeof(MNISTData));
    int image_count = 0, label_count = 0;
    data->images = _load_mnist_images(images_path, &image_count);
    data->labels = _load_mnist_labels(labels_path, &label_count);
    assert(image_count == label_count);
    data->count = image_count;
    return data;
}

void free_mnist_data(MNISTData *data) {
    for (int i = 0; i < data->count; i++) {
        tensor_free(data->images[i]);
        vector_free(data->labels[i]);
    }
    free(data);
}