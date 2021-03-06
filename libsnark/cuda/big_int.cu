#include <cuda.h>
#include <stdint.h>
#include <libsnark/cuda/big_int.hpp>

// r = a + b
__device__ void big_int_add(big_int r, big_int a, big_int b) {
  asm ("add.cc.u32 %0, %1, %2;": "=r"(r[0]): "r"(a[0]), "r"(b[0]));
  for (int i = 1; i < GPU_N_LIMBS; i++) {
    asm ("addc.cc.u32 %0, %1, %2;": "=r"(r[i]): "r"(a[i]), "r"(b[i]));
  }
}

// r = a - b
__device__ void big_int_sub(big_int r, big_int a, big_int b) {
  asm ("sub.cc.u32 %0, %1, %2;": "=r"(r[0]): "r"(a[0]), "r"(b[0]));
  for (int i = 1; i < GPU_N_LIMBS; i++) {
    asm ("subc.cc.u32 %0, %1, %2;": "=r"(r[i]): "r"(a[i]), "r"(b[i]));
  }
}

// test bit i of r
__device__ int big_int_test(big_int r, int i) {
  return (r[i / 32] >> (i % 32)) & 0x00000001;
}

// a < b
__device__ int big_int_lt(big_int a, big_int b) {
  int lt = 0;
  int eq = 1;
  for (int i = GPU_N_LIMBS - 1; i >= 0; i--) {
    if (a[i] < b[i]) {
      lt |= eq;
    }
    if (a[i] > b[i]) {
      eq = 0;
    }
  }

  return lt;
}

__device__ int big_int_eq(big_int a, big_int b) {
  int eq = 1;
  for (int i = 0; i < GPU_N_LIMBS; i++) {
    eq = eq && (a[i] == b[i]);
  }
  return eq;
}

__device__ int big_int_is_zero(big_int a) {
  int is_zero = 1;
  for (int i = 0; i < GPU_N_LIMBS; i++) {
    is_zero = is_zero && (a[i] == 0);
  }
  return is_zero;
}

// r = a * b * R^-1 mod m
// implements CIOS montgomery multiplication for R = 2^(32 * GPU_N_LIMBS)
// m is the modulus; inv_m should be -1/m[0] (mod 2^32)
__device__ void big_int_mulred(big_int r, big_int a, big_int b, big_int m, uint32_t inv_m) {
  uint64_t overflow; // effectively limb GPU_N_LIMBS and (GPU_N_LIMBS + 1) of r
  for (int i = 0; i < GPU_N_LIMBS; i++) {
    r[i] = 0;
  }
  for (int i = 0; i < GPU_N_LIMBS; i++) {
    uint64_t x = 0;
    for (int j = 0; j < GPU_N_LIMBS; j++) {
      x = (uint64_t) a[j] * b[i] + (x >> 32) + r[j];
      r[j] = x & 0xffffffff;
    }

    overflow = (overflow & 0xffffffff) + (x >> 32);

    uint32_t k = r[0] * inv_m;
    x = (uint64_t) k * m[0] + r[0];
    for (int j = 1; j < GPU_N_LIMBS; j++) {
      x = (uint64_t) k * m[j] + r[j] + (x >> 32);
      r[j - 1] = x & 0xffffffff;
    }

    x = (overflow & 0xffffffff) + (x >> 32);
    r[GPU_N_LIMBS - 1] = (x & 0xffffffff);
    overflow = ((overflow >> 32) + (x >> 32)) & 0xffffffff;
  }
}
