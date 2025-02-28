#include <assert.h>
#include "matrix.h"

void _matrix_destory(void *obj){
    Matrix *matrix = (Matrix *)obj;
    for (int i = 0; i < matrix->rows; i++) {
        vector_free(matrix->vectors[i]);
    }
    free(matrix->vectors);
    free(matrix);
}

Matrix *_matrix_create_empty(int rows, int cols) {
    Matrix *matrix = (Matrix *)malloc(sizeof(Matrix));
    matrix->rows = rows;
    matrix->cols = cols;
    matrix->vectors = (Vector **)malloc(rows * sizeof(Vector *));
    refer_init(&matrix->refer);
    return matrix;
}

Matrix *matrix_create(int rows, int cols) {
    Matrix *matrix = _matrix_create_empty(rows, cols);
    for (int i = 0; i < rows; i++) {
        matrix->vectors[i] = vector_create(cols);
    }
    return matrix;
}

void matrix_free(Matrix *matrix) {
    unrefer(matrix, _matrix_destory);
}

Matrix *matrix_create_from_array(float* data, int rows, int cols) {
    Matrix *matrix = matrix_create(rows, cols);
    for (int i = 0; i < rows; i++) {
        matrix->vectors[i] = vector_create_from_array(data + i * cols, cols);
    }
    return matrix;
}

Matrix *matrix_copy(Matrix *matrix) {
    Matrix *copy = _matrix_create_empty(matrix->rows, matrix->cols);
    for (int i = 0; i < matrix->rows; i++) {
        copy->vectors[i] = vector_copy(matrix->vectors[i]);
    }
    return copy;
}

float matrix_get(Matrix *matrix, int row, int col) {
    return vector_get(matrix->vectors[row], col);
}

void matrix_set(Matrix *matrix, int row, int col, float value) {
    vector_set(matrix->vectors[row], col, value);
}

Matrix *matrix_reshape(Matrix *matrix, int rows, int cols) {
    assert(matrix->rows * matrix->cols == rows * cols);

    Matrix *reshaped = matrix_create(rows, cols);
    int index = 0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            index = i * cols + j;
            matrix_set(reshaped, i, j, matrix_get(matrix, index / matrix->cols, index % matrix->cols));
        }
    }
    return reshaped;
}

Matrix *matrix_transpose(Matrix *matrix) {
    Matrix *transposed = matrix_create(matrix->cols, matrix->rows);
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->cols; j++) {
            matrix_set(transposed, j, i, matrix_get(matrix, i, j));
        }
    }
    return transposed;
}

Matrix *matrix_multiply(Matrix *matrix, Matrix *other) {
    assert(matrix->cols == other->rows);

    Matrix *result = matrix_create(matrix->rows, other->cols);
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            float sum = 0.0f;
            for (int k = 0; k < matrix->cols; k++) {
                sum += matrix_get(matrix, i, k) * matrix_get(other, k, j);
            }
            matrix_set(result, i, j, sum);
        }
    }
    return result;
}
