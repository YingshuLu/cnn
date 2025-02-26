#include "refer.h"

void refer_init(Refer *refer) {
    atomic_init(&refer->count, 1);
}

void refer(void *obj) {
    Refer *refer = (Refer *)obj;
    atomic_fetch_add(&refer->count, 1);
}

void unrefer(void *obj, void (*destroy)(void *)) {
    Refer *refer = (Refer *)obj;
    if (refer && atomic_fetch_sub(&refer->count, 1) == 1) {
        if (destroy) {
            destroy(obj);
        } else {
            free(obj);
        }
    }
}