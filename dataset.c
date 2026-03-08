#include "dataset.h"
#include "rng.h"

#include <stdio.h>
#include <memory.h>
#include <assert.h>


u0 inspect(f32 *feature, f32 *label)
{
    for(i32 i=0; i<28; ++i)
    {
        for(i32 j=0; j<28; ++j)
        {
            f32 val=feature[i*28+j];
            u32 color=232+(u32)(val*23);
            printf("\x1b[48;5;%dm  ", color);
        }
        printf("\n");
    }

    printf("\x1b[0m");

    for(i32 i=0; i<10; ++i) printf("%.0f ", label[i]);
    printf("\n\n");
}

Dataset *dataset_load(
    struct Arena *arena,
    usz num_samples,
    usz feature_size,
    usz label_size,
    const char *feature_file,
    const char *label_file
)
{
    Dataset *ds=PUSH_STRUCT(arena, Dataset);

    ds->num_samples=num_samples;
    ds->feature_size=feature_size;
    ds->label_size=label_size;
    ds->order=PUSH_ARRAY_NZ(arena, usz, num_samples);
    for(usz i=0; i<num_samples; ++i) ds->order[i]=i;
    ds->feature=matrix_load(arena, num_samples, feature_size, feature_file);
    ds->label=matrix_create(arena, num_samples, label_size);

    ArenaTemp scratch=arena_get_scratch(NULL, 0);
    struct Matrix *label=matrix_load(scratch.arena, num_samples, 1, label_file);

    for(usz i=0; i<num_samples; ++i)
    {
        usz val=label->data[i];
        ds->label->data[i*label_size+val]=1.0f;
    }

    inspect(ds->feature->data, ds->label->data);

    arena_release_scratch(scratch);

    return ds;
}

u0 dataset_shuffle(Dataset *ds)
{
    for(usz i=ds->num_samples-1; i>0; --i)
    {
        usz j=rng_rand()%(i+1);

        usz tmp=ds->order[j];
        ds->order[j]=ds->order[i];
        ds->order[i]=tmp;
    }
}

u0 dataset_get(struct Matrix *feature, struct Matrix *label, Dataset *ds, usz idx, usz n)
{
    assert(idx+n-1<ds->num_samples);

    for(usz i=0; i<n; ++i)
    {
        usz p=ds->order[idx+i];
        memcpy(
            feature->data+i*ds->feature_size,
            ds->feature->data+p*ds->feature_size,
            sizeof(f32)*ds->feature_size
        );
        memcpy(
            label->data+i*ds->label_size,
            ds->label->data+p*ds->label_size,
            sizeof(f32)*ds->label_size
        );
    }
}
