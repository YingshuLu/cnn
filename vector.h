#ifndef VECTOR_H
#define VECTOR_H

#include <stdlib.h>
#include "refer.h"

#define vector_refer(vector) (refer(&(vector)->refer), (vector))
#define vector_unrefer(vector) vector_free(vector)

typedef struct Vector {
    Refer refer;
    float *data;
    int size;
} Vector;

Vector* vector_create(int size);
void vector_free(Vector *vector);
Vector* vector_create_from_array(float *array, int size);
int vector_size(Vector *vector);
void vector_set(Vector *vector, int index, float value);
float vector_get(Vector *vector, int index);
Vector* vector_copy(Vector *vector);
void vector_fill(Vector *vector, float value);
void vector_randomize(Vector *vector, float min, float max);
void vector_add_value(Vector *vector, float value);
void vector_sub_value(Vector *vector, float value);
void vector_mul_value(Vector *vector, float value);
void vector_div_value(Vector *vector, float value);
void vector_mod_value(Vector *vector, float value);
void vector_pow_value(Vector *vector, float value);
void vector_add(Vector *vector, Vector *other);
void vector_sub(Vector *vector, Vector *other);
void vector_multiply(Vector *vector, Vector *other);
float vector_dot(Vector *vector, Vector *other);
void vector_normalize(Vector *vector);

#endif
