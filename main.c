#include "trainer.h"

#include <stdio.h>
#include <memory.h>


i32 main(u0)
{
    struct Arena *arena=arena_create(GB(1), MB(1));

    struct Trainer *trainer=trainer_create(arena, 784, 10);

    memcpy(trainer->model->input->val->data, trainer->test->feature->data, sizeof(f32) * trainer->batch_size * trainer->feature_size);
    model_forward(trainer->model);

    printf("Pre-training output:\n");
    for(usz i=0; i<trainer->batch_size; ++i)
    {
        for(usz j=0; j<trainer->label_size; j++)
            printf("%.2f ", trainer->model->output->val->data[i*trainer->label_size+j]);
        printf("\n");
    }
    printf("\n");

    trainer_train(trainer);

    memcpy(trainer->model->input->val->data, trainer->test->feature->data, sizeof(f32) * trainer->batch_size * trainer->feature_size);
    model_forward(trainer->model);

    printf("Post-training output:\n");
    for(usz i=0; i<trainer->batch_size; ++i)
    {
        for(usz j=0; j<trainer->label_size; j++)
            printf("%.2f ", trainer->model->output->val->data[i*trainer->label_size+j]);
        printf("\n");
    }
    printf("\n");

    return 0;
}
