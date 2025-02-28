#ifndef POOL_LAYER_H
#define POOL_LAYER_H

#include "tensor_layer.h"
typedef struct PoolLayer {
    TensorLayer layer;
    int pool_size;
    int stride;
    int padding;
} PoolLayer;


#endif
