#ifndef MATRIX_H
#define MATRIX_H

#include "refer.h"
#include "vector.h"

#define matrix_refer(matrix) (refer(&(matrix)->refer), (matrix))
#define matrix_unrefer(matrix) matrix_free(matrix)

typedef struct Matrix {
    Refer refer;
    int rows;
    int cols;
    Vector **vectors;
} Matrix;

Matrix *matrix_create(int rows, int cols);
void matrix_free(Matrix *matrix);
Matrix *matrix_create_from_array(float* data, int rows, int cols);
Matrix *matrix_copy(Matrix *matrix);
float matrix_get(Matrix *matrix, int row, int col);
void matrix_set(Matrix *matrix, int row, int col, float value);
Matrix *matrix_reshape(Matrix *matrix, int rows, int cols);
Matrix *matrix_transpose(Matrix *matrix);

Matrix *matrix_multiply(Matrix *matrix, Matrix *other);

#endif