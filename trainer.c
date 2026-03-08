#include "trainer.h"
#include "layer.h"
#include "loss.h"
#include "metric.h"

#include <stdio.h>
#include <stddef.h>
#include <memory.h>


Trainer *trainer_create(Arena *arena, usz feature_size, usz label_size)
{
    Trainer *trainer=PUSH_STRUCT(arena, Trainer);

    trainer->epochs=10;
    trainer->batch_size=8;
    trainer->learning_rate=0.01f;

    trainer->feature_size=feature_size;
    trainer->label_size=label_size;

    trainer->train=dataset_load(arena, 60000, feature_size, label_size, "data/train_images.mat", "data/train_labels.mat");
    trainer->val=NULL;
    trainer->test=dataset_load(arena, 10000, feature_size, label_size, "data/test_images.mat", "data/test_labels.mat");

    trainer->model=PUSH_STRUCT(arena, struct Model);
    trainer->model->num_nodes=0;
    trainer_create_model(arena, trainer);
    model_compile(arena, trainer->model);

    return trainer;
}

u0 trainer_create_model(Arena *arena, Trainer *trainer)
{
    Model *model=trainer->model;

    model->input=layer_create_input(arena, &model->num_nodes, trainer->batch_size, trainer->feature_size);
    model->target=layer_create_target(arena, &model->num_nodes, trainer->batch_size, trainer->label_size);

    ExecNode *input=model->input;

    input=layer_create_linear(arena, &model->num_nodes, input, trainer->feature_size, 16);
    input=layer_create_relu(arena, &model->num_nodes, input);

    input=layer_create_linear(arena, &model->num_nodes, input, 16, 16);
    input=layer_create_relu(arena, &model->num_nodes, input);
    // struct ExecNode *a1 = model_add_layer_add(arena, trainer->model, input, z1_c, 0);

    model->output=layer_create_linear(arena, &model->num_nodes, input, 16, trainer->label_size);
    model->output->flags|=EXEC_FLAG_OUTPUT;

    model->cost=loss_create_cross_entropy_with_logits(arena, &model->num_nodes, trainer->model->output, trainer->model->target);
}

u0 trainer_train(Trainer *trainer)
{
    Model *model=trainer->model;
    Dataset *train=trainer->train;
    Dataset *test=trainer->test;

    usz num_batches=train->num_samples/trainer->batch_size;

    for(usz epoch=0; epoch<trainer->epochs; ++epoch)
    {
        dataset_shuffle(train);
        for(usz batch=0; batch<num_batches; ++batch)
        {
            dataset_get(
                model->input->val,
                model->target->val,
                train,
                batch*trainer->batch_size,
                trainer->batch_size
            );

            model_forward(model);
            model->cost->forward(model->cost);

            model_zero_grad(model);

            model->cost->backward(model->cost);
            model_backward(model);

            model_update(model, trainer->learning_rate, trainer->batch_size);

            printf(
                "Epoch %2lu / %2lu, Batch %4lu / %4lu, Average Cost: %.4f\r",
                epoch + 1, trainer->epochs,
                batch + 1, num_batches, model->cost->val->data[0]
            );
            fflush(stdout);

        }

        printf("\n");

        usz num_correct=0;
        f32 total_cost=0.0f;
        for(usz i=0; i+trainer->batch_size<test->num_samples; i+=trainer->batch_size)
        {
            dataset_get(
                model->input->val,
                model->target->val,
                test,
                i,
                trainer->batch_size
            );

            model_forward(model);
            model->cost->forward(model->cost);

            total_cost+=model->cost->val->data[0]*trainer->batch_size;
            num_correct+=accuracy(model->output->val, model->target->val);
        }

        f32 avg_cost=total_cost/(f32)test->num_samples;
        printf(
            "Test Completed! Accuracy: %5lu/%5lu (%.1f%%) | Average Cost: %.4f\n",
            num_correct, test->num_samples, (f32)num_correct / test->num_samples* 100.0f,
            avg_cost
        );
    }
}

