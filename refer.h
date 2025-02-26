#ifndef REFER_H
#define REFER_H

#include <stdatomic.h>
#include <stdlib.h>

typedef struct Refer {
    atomic_int count;
} Refer;

void refer_init(Refer *refer);
void refer(void *obj);
void unrefer(void *obj, void (*destroy)(void *));

#endif
