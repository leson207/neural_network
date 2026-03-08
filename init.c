#include "init.h"

#include "math.h"


u0 init_random(Matrix *matrix)
{
    f32 bound=sqrtf(6.0f / (matrix->rows+ matrix->cols));
    matrix_fill_random(matrix, -bound, bound);
}
