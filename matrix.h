#pragma once

#include "base.h"
#include "arena.h"


typedef struct Matrix Matrix;

struct Matrix
{
    usz rows;
    usz cols;
    usz size;

    f32 *data;
};

Matrix *matrix_create(Arena *arena, usz rows, usz cols);
Matrix *matrix_load(Arena *arena, usz rows, usz cols, const char *filename);

u0 matrix_fill_constant(Matrix *out, f32 value);
u0 matrix_fill_random(Matrix *out, f32 low, f32 high);

u0 matrix_clear(Matrix *out);
u0 matrix_scale(Matrix *out, f32 scale);

f32 matrix_sum(Matrix *matrix);
usz matrix_argmax(Matrix *matrix);

u0 matrix_relu(Matrix *out, const Matrix *in);
u0 matrix_softmax(Matrix *out, const Matrix *in);

u0 matrix_add(Matrix *out, Matrix *a, Matrix *b, b8 broadcast);
u0 matrix_sub(Matrix *out, Matrix *a, Matrix *b, b8 broadcast);
u0 matrix_mul(Matrix *out, Matrix *a, Matrix *b, b8 zero_out, b8 transpose_a, b8 transpose_b);
u0 matrix_cross_entropy(Matrix *out, const Matrix *a, const Matrix *b);
