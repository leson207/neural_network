#include "arena.h"

#include <string.h>


struct Arena
{
    u64 reserve_size;
    u64 commit_size;

    u64 pos;
    u64 commit_pos;
};

Arena *arena_create(u64 reserve_size, u64 commit_size)
{
    u32 pagesize=get_pagesize();

    reserve_size=ALIGN_UP_POW2(reserve_size, pagesize);
    commit_size=ALIGN_UP_POW2(commit_size, pagesize);

    struct Arena *arena=mem_reserve(reserve_size);

    if(!mem_commit(arena, commit_size)) return NULL;

    arena->reserve_size=reserve_size;
    arena->commit_size=commit_size;
    arena->pos=ARENA_BASE_POS;
    arena->commit_pos=commit_size;

    return arena;
}

u0 *arena_push(Arena *arena, u64 size, b32 non_zero)
{
    u64 pos_aligned=ALIGN_UP_POW2(arena->pos, ARENA_ALIGN);
    u64 new_pos=pos_aligned+size;

    if(new_pos>arena->reserve_size) return NULL;

    if(new_pos>arena->commit_pos)
    {
        u64 new_commit_pos=new_pos;
        new_commit_pos+=arena->commit_size-1;
        new_commit_pos-=new_commit_pos%arena->commit_size;
        new_commit_pos=MIN(new_commit_pos, arena->reserve_size);

        u8 *mem=(u8 *)arena+arena->commit_pos;
        u64 commit_size=new_commit_pos-arena->commit_pos;

        if(!mem_commit(mem, commit_size)) return NULL;

        arena->commit_pos=new_commit_pos;
    }

    arena->pos=new_pos;

    u8 *out=(u8 *)arena+pos_aligned;

    if(!non_zero) memset(out, 0, size);

    return out;
}

u0 arena_pop(Arena *arena, u64 size)
{
    size=MIN(size, arena->pos - ARENA_BASE_POS);
    arena->pos-=size;

    return;
}

u0 arena_pop_to(Arena *arena, u64 pos)
{
    u64 size=pos<arena->pos ? arena->pos-pos : 0;
    arena_pop(arena, size);

    return;
}

u0 arena_clear(struct Arena *arena)
{
    arena_pop_to(arena, ARENA_BASE_POS);

    return;
}

ArenaTemp arena_temp_begin(Arena *arena)
{
    return (ArenaTemp){
        .arena = arena,
        .start_pos = arena->pos
    };
}

u0 arena_temp_end(ArenaTemp temp)
{
    arena_pop_to(temp.arena, temp.start_pos);

    return;
}

static __thread Arena *_scratch_arenas[2] = { NULL, NULL };

ArenaTemp arena_get_scratch(Arena **conflicts, u32 num_conflicts)
{
    i32 scratch_index=-1;

    for(i32 i=0; i<2; ++i)
    {
        b32 conflict_found=false;

        for(u32 j=0; j<num_conflicts; ++j)
        {
            if(_scratch_arenas[i]==conflicts[j])
            {
                conflict_found=true;
                break;
            }
        }

        if(!conflict_found)
        {
            scratch_index=i;
            break;
        }
    }

    if(scratch_index==-1) return (ArenaTemp){0};

    Arena **selected=&_scratch_arenas[scratch_index];

    if(*selected==NULL) *selected=arena_create(MB(64), MB(1));

    return arena_temp_begin(*selected);
}

u0 arena_release_scratch(ArenaTemp scratch)
{
    arena_temp_end(scratch);

    return;
}

#ifndef _DEFAULT_SOURCE
#define _DEFAULT_SOURCE
#endif

#include <unistd.h>
#include <sys/mman.h>

u32 get_pagesize(u0)
{
    return (u32)sysconf(_SC_PAGESIZE);
}

u0 *mem_reserve(u64 size)
{
    u0 *out = mmap(NULL, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if(out==MAP_FAILED) return NULL;

    return out;
}

b32 mem_commit(u0 *ptr, u64 size)
{
    return !mprotect(ptr, size, PROT_READ | PROT_WRITE);
}

b32 mem_decommit(u0 *ptr, u64 size)
{
    if(mprotect(ptr, size, PROT_NONE)) return false;

    return !madvise(ptr, size, MADV_DONTNEED);
}

b32 mem_release(u0 *ptr, u64 size)
{
    return !munmap(ptr, size);
}
