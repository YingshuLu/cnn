#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include "operator.h"
#include "vector.h"

typedef float(*_value_operation)(float target, float value);

void _vector_value_operation(Vector *vector, float value, _value_operation operation) {
    for (int i = 0; i < vector->size; i++) {
        vector->data[i] = operation(vector->data[i], value);
    }
}

void _vector_vector_operation(Vector *vector, Vector *other, _value_operation operation) {
    assert(vector->size == other->size);
    for (int i = 0; i < vector->size; i++) {
        vector->data[i] = operation(vector->data[i], other->data[i]);
    }
}

void _vector_destroy(void *object) {
    Vector *vector = (Vector *)object;
    free(vector->data);
    free(vector);
}

Vector *vector_create(int size) {
    Vector *vector = (Vector *)malloc(sizeof(Vector));
    vector->data = (float *)calloc(size, sizeof(float));
    vector->size = size;
    refer_init(&vector->refer);
    return vector;
}

void vector_free(Vector *vector) {
    if (!vector) {
        return;
    }
    unrefer(vector, _vector_destroy);
}

Vector *vector_create_from_array(float *array, int size) {
    Vector *vector = vector_create(size);
    for (int i = 0; i < size; i++) {
        vector->data[i] = array[i];
    }
    return vector;
}

int vector_size(Vector *vector) {
    return vector->size;
}

void vector_set(Vector *vector, int index, float value) {
    vector->data[index] = value;
}

float vector_get(Vector *vector, int index) {
    return vector->data[index];
}

float vector_sum(Vector *vector) {
    float sum = 0;
    for (int i = 0; i < vector->size; i++) {
        sum += vector->data[i];
    }
    return sum;
}

void vector_apply(Vector *vector, float (*operation)(float)) {
    for (int i = 0; i < vector->size; i++) {
        vector->data[i] = operation(vector->data[i]);
    }
}

Vector *vector_copy(Vector *vector) {
    Vector *copy = vector_create(vector->size);
    vector_add(copy, vector);
    return copy;
}

void vector_fill(Vector *vector, float value) {
    _vector_value_operation(vector, value, float_fill);
}

void vector_randomize(Vector *vector, float min, float max) {
    for (int i = 0; i < vector->size; i++) {
        vector->data[i] = min + (float)rand() / RAND_MAX * (max - min);
    }
}

void vector_add_value(Vector *vector, float value) {
    _vector_value_operation(vector, value, float_add);
}

void vector_sub_value(Vector *vector, float value) {
    _vector_value_operation(vector, value, float_sub);
}

void vector_mul_value(Vector *vector, float value) {
    _vector_value_operation(vector, value, float_mul);
}

void vector_div_value(Vector *vector, float value) {
    _vector_value_operation(vector, value, float_div);
}

void vector_mod_value(Vector *vector, float value) {
    _vector_value_operation(vector, value, float_mod);
}

void vector_pow_value(Vector *vector, float value) {
    _vector_value_operation(vector, value, float_pow);
}

void vector_add(Vector *vector, Vector *other) {
    _vector_vector_operation(vector, other, float_add);
}

void vector_sub(Vector *vector, Vector *other) {
    _vector_vector_operation(vector, other, float_sub);
}

float vector_dot(Vector *vector, Vector *other) {
    assert(vector->size == other->size);
    float result = 0;
    for (int i = 0; i < vector->size; i++) {
        result += vector->data[i] * other->data[i];
    }
    return result;
}

void vector_multiply(Vector *vector, Vector *other) {
    assert(vector->size == other->size);
    for (int i = 0; i < vector->size; i++) {
        vector->data[i] = vector->data[i] * other->data[i];
    }
}

void vector_normalize(Vector *vector) {
    float length = 0;
    for (int i = 0; i < vector->size; i++) {
        length += vector->data[i] * vector->data[i];
    }
    length = sqrt(length);
    if (length == 0) {
        return;
    }
    vector_div_value(vector, length);
}
