#pragma once

#include "base.h"

#include <stdbool.h>


#define ARENA_BASE_POS (sizeof(Arena))
#define ARENA_ALIGN (sizeof(u0 *))

typedef struct Arena Arena;

Arena *arena_create(u64 reserve_size, u64 commit_size);
u0 *arena_push(Arena *arena, u64 size, b32 non_zero);
u0 arena_pop(Arena *arena, u64 size);
u0 arena_pop_to(Arena *arena, u64 pos);
// u0 arena_clear(struct Arena *arena);

typedef struct ArenaTemp ArenaTemp;
struct ArenaTemp
{
    struct Arena *arena;
    u64 start_pos;
};

ArenaTemp arena_temp_begin(Arena *arena);
u0 arena_temp_end(ArenaTemp temp);
ArenaTemp arena_get_scratch(Arena **conflicts, u32 num_conflicts);
u0 arena_release_scratch(ArenaTemp scratch);

#define PUSH_STRUCT(arena, T) (T *)arena_push((arena), sizeof(T), false)
#define PUSH_STRUCT_NZ(arena, T) (T*)arena_push((arena), sizeof(T), true)
#define PUSH_ARRAY(arena, T, n) (T *)arena_push((arena), sizeof(T) * (n), false)
#define PUSH_ARRAY_NZ(arena, T, n) (T*)arena_push((arena), sizeof(T) * (n), true)

u32 get_pagesize(u0);
u0 *mem_reserve(u64 size);
b32 mem_commit(u0 *ptr, u64 size);
// b32 mem_decommit(u0 *ptr, u64 size);
// b32 mem_release(u0 *ptr, u64 size);
