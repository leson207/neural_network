#pragma once

#include <stdint.h>
#include <sys/types.h>

typedef int8_t i8;
typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;
typedef ssize_t isz;

typedef void u0;
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef size_t usz;

typedef float f32;
typedef double f64;

typedef i8 b8;
typedef i32 b32;

#define KB(n) ((u64)(n)<<10)
#define MB(n) ((u64)(n)<<20)
#define GB(n) ((u64)(n)<<30)

#define MIN(a, b) (((a)<(b)) ? (a) : (b))
#define MAX(a, b) (((a)>(b)) ? (a) : (b))
#define SWAP(a, b)          \
    do {                     \
        __typeof__(a) tmp = (a); \
        (a) = (b);           \
        (b) = tmp;           \
    } while(0)
#define ALIGN_UP_POW2(n, p) (((u64)(n) + ((u64)(p) - 1)) & (~((u64)(p) - 1)))
