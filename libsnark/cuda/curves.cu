#include <stdio.h>
#include <stdint.h>

#include <libsnark/cuda/curves.hpp>
#include <libsnark/cuda/big_int.cu>

#include <cuda_runtime.h>

#define BLOCK_SIZE 1024

void cuda_check_error() {
  cudaError_t err = cudaPeekAtLastError();
  if (err) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
    std::exit(0);
  }
}

struct cuda_mnt4_G1 {
  big_int X;
  big_int Y;
  big_int Z;
};

// a trick to avoid defining things twice
#ifdef __CUDA_ARCH__
#define CONSTANT __constant__
#else
#define CONSTANT
#endif

CONSTANT cuda_mnt4_G1 cuda_mnt4_G1_zero = {
  .X = {},
  // 1 in montgomery representation
  .Y = {0x5863845c, 0x18c31a7b, 0xe3b68df5, 0xe9de7a15, 0x28faab40, 0xc5df8587, 0x647b5197, 0x29184098, 0x223d33c3, 0x000001c1},
  .Z = {}
};

CONSTANT big_int mnt4_Fq_modulus = {0x71660001, 0xc90cd65a, 0x51200e12, 0x41a9e35e, 0x5d1330ea, 0xcaeec963, 0xa7b0548e, 0xa266249d, 0xf7bcd473, 0x000003bc};
CONSTANT big_int mnt4_Fq_R2 = {0x5613d220, 0x0065acec, 0xbf2bc893, 0xa266a1ad, 0x318850e1, 0x66bd7673, 0xad38d47b, 0x1f32e014, 0xf0918a34, 0x00000224};
CONSTANT uint32_t mnt4_Fq_inv = 0x7165ffff;

// 2 in montgomery representation (mod mnt4_Fq_modulus)
CONSTANT big_int mnt4_G1_coeff_a = {0xb0c708b8, 0x318634f6, 0xc76d1bea, 0xd3bcf42b, 0x51f55681, 0x8bbf0b0e, 0xc8f6a32f, 0x52308130, 0x447a6786, 0x00000382};
// 4 in montgomery representation
CONSTANT big_int mnt4_G1_coeff_4 = {0xf028116f, 0x99ff9392, 0x3dba29c1, 0x65d004f9, 0x46d77c19, 0x4c8f4cb9, 0xea3cf1d0, 0x01faddc3, 0x9137fa99, 0x00000347};

__device__ void cuda_mnt4_Fq_add(big_int r, big_int a, big_int b) {
  big_int_add(r, a, b);
  if (!big_int_lt(r, mnt4_Fq_modulus)) {
    big_int_sub(r, r, mnt4_Fq_modulus);
  }
}

__device__ void cuda_mnt4_Fq_sub(big_int r, big_int a, big_int b) {
  // not the most efficient
  big_int_add(r, a, mnt4_Fq_modulus);
  big_int_sub(r, r, b);
  if (!big_int_lt(r, mnt4_Fq_modulus)) {
    big_int_sub(r, r, mnt4_Fq_modulus);
  }
}

// assume inputs in montgomery representation
__device__ void cuda_mnt4_Fq_mul(big_int r, big_int a, big_int b) {
  big_int_mulred(r, a, b, mnt4_Fq_modulus, mnt4_Fq_inv);
}

__device__ int cuda_mnt4_G1_is_zero(cuda_mnt4_G1 *a) {
  return big_int_is_zero(a->X) && big_int_is_zero(a->Z);
}

__device__ void cuda_mnt4_G1_dbl(cuda_mnt4_G1 *r, cuda_mnt4_G1 *a) {
  if (cuda_mnt4_G1_is_zero(a)) {
    for (int i = 0; i < GPU_N_LIMBS; i++) {
      r->X[i] = a->X[i];
      r->Y[i] = a->Y[i];
      r->Z[i] = a->Z[i];
    }
  }

  big_int B, R, RR, XX, XX2, ZZ, h, s, ss, sss, t0, t1, t10, t2, t3, t4, t5, t6, t7, t8, t9, w;
  cuda_mnt4_Fq_mul(XX, a->X, a->X);
  cuda_mnt4_Fq_mul(ZZ, a->Z, a->Z);
  cuda_mnt4_Fq_add(XX2, XX, XX);
  cuda_mnt4_Fq_add(t0, XX, XX2);
  cuda_mnt4_Fq_add(t1, ZZ, ZZ);
  cuda_mnt4_Fq_add(w, t1, t0);
  cuda_mnt4_Fq_mul(t2, a->Y, a->Z);
  cuda_mnt4_Fq_add(s, t2, t2);
  cuda_mnt4_Fq_mul(ss, s, s);
  cuda_mnt4_Fq_mul(sss, s, ss);
  cuda_mnt4_Fq_mul(R, a->Y, s);
  cuda_mnt4_Fq_mul(RR, R, R);
  cuda_mnt4_Fq_add(t3, a->X, R);
  cuda_mnt4_Fq_mul(t4, t3, t3);
  cuda_mnt4_Fq_sub(t5, t4, XX);
  cuda_mnt4_Fq_sub(B, t5, RR);
  cuda_mnt4_Fq_mul(t6, w, w);
  cuda_mnt4_Fq_add(t7, B, B);
  cuda_mnt4_Fq_sub(h, t6, t7);
  cuda_mnt4_Fq_mul(r->X, h, s);
  cuda_mnt4_Fq_sub(t8, B, h);
  cuda_mnt4_Fq_add(t9, RR, RR);
  cuda_mnt4_Fq_mul(t10, w, t8);
  cuda_mnt4_Fq_sub(r->Y, t10, t9);
  cuda_mnt4_Fq_sub(r->Z, t10, t9);
}

__device__ void cuda_mnt4_G1_add(cuda_mnt4_G1 *r, cuda_mnt4_G1 *a, cuda_mnt4_G1 *b) {
  if (cuda_mnt4_G1_is_zero(a)) {
    for (int i = 0; i < GPU_N_LIMBS; i++) {
      r->X[i] = b->X[i];
      r->Y[i] = b->Y[i];
      r->Z[i] = b->Z[i];
    }
  } else if (cuda_mnt4_G1_is_zero(b)) {
    for (int i = 0; i < GPU_N_LIMBS; i++) {
      r->X[i] = a->X[i];
      r->Y[i] = a->Y[i];
      r->Z[i] = a->Z[i];
    }
  }

  big_int F, G, L, LL, M, R, S1, S2, T, TT, U1, U2, W, ZZ, t0, t1, t10, t11, t12, t13, t14, t15, t16, t2, t3, t4, t5, t6, t7, t8, t9;
  cuda_mnt4_Fq_mul(U1, a->X, b->Z);
  cuda_mnt4_Fq_mul(U2, b->X, a->Z);
  cuda_mnt4_Fq_mul(S1, a->Y, b->Z);
  cuda_mnt4_Fq_mul(S2, b->Y, a->Z);
  cuda_mnt4_Fq_mul(ZZ, a->Z, b->Z);
  cuda_mnt4_Fq_add(T, U1, U2);
  cuda_mnt4_Fq_mul(TT, T, T);
  cuda_mnt4_Fq_add(M, S1, S2);
  cuda_mnt4_Fq_mul(t0, ZZ, ZZ);
  cuda_mnt4_Fq_add(t1, t0, t0);
  cuda_mnt4_Fq_mul(t2, U1, U2);
  cuda_mnt4_Fq_sub(t3, TT, t2);
  cuda_mnt4_Fq_add(R, t3, t1);
  cuda_mnt4_Fq_mul(F, ZZ, M);
  cuda_mnt4_Fq_mul(L, M, F);
  cuda_mnt4_Fq_mul(LL, L, L);
  cuda_mnt4_Fq_add(t4, T, L);
  cuda_mnt4_Fq_mul(t5, t4, t4);
  cuda_mnt4_Fq_sub(t6, t5, TT);
  cuda_mnt4_Fq_sub(G, t6, LL);
  cuda_mnt4_Fq_mul(t7, R, R);
  cuda_mnt4_Fq_add(t8, t7, t7);
  cuda_mnt4_Fq_sub(W, t8, G);
  cuda_mnt4_Fq_mul(t9, F, W);
  cuda_mnt4_Fq_add(r->X, t9, t9);
  cuda_mnt4_Fq_add(t10, W, W);
  cuda_mnt4_Fq_sub(t11, G, t10);
  cuda_mnt4_Fq_add(t12, LL, LL);
  cuda_mnt4_Fq_mul(t13, R, t11);
  cuda_mnt4_Fq_sub(r->Y, t13, t12);
  cuda_mnt4_Fq_mul(t14, F, F);
  cuda_mnt4_Fq_mul(t15, F, t14);
  cuda_mnt4_Fq_add(t16, t15, t15);
  cuda_mnt4_Fq_add(r->Z, t16, t16);
}

// r = a * s
__device__ void cuda_mnt4_G1_mul(cuda_mnt4_G1 *r, cuda_mnt4_G1 *a, big_int s) {
  for (int i = 0; i < GPU_N_LIMBS; i++) {
    r->X[i] = cuda_mnt4_G1_zero.X[i];
    r->Y[i] = cuda_mnt4_G1_zero.Y[i];
    r->Z[i] = cuda_mnt4_G1_zero.Z[i];
  }

  for (int i = 32 * GPU_N_LIMBS - 1; i >= 0; i--) {
    cuda_mnt4_G1_dbl(r, r);
    if (big_int_test(s, i)) {
      cuda_mnt4_G1_add(r, r, a);
    }
  }
}

// d_res[i] = d_vec[i]^d_scalar[i]
__global__ void mnt4_G1_multi_exp_kernel(int len, cuda_mnt4_G1 *d_res, cuda_mnt4_G1 *d_vec, big_int *d_scalar) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    cuda_mnt4_G1_mul(&d_res[i], &d_vec[i], d_scalar[i]);
  }
}

// d_res = sum(d_vec) (in fact blockwise, so d_vec[i] is the sum of the ith block)
__global__ void mnt4_G1_reduce_kernel(int len, cuda_mnt4_G1 *d_res, cuda_mnt4_G1 *d_vec) {
  extern __shared__ cuda_mnt4_G1 data[];

  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < len) {
    data[threadIdx.x] = d_vec[i];

    __syncthreads();

    for (int s = 1; s < blockDim.x; s *= 2) {
      if (i % (2 * s) == 0) {
        cuda_mnt4_G1_add(&data[i], &data[i], &data[i + s]);
      }
      __syncthreads();
    }

    if (i == 0) {
      d_res[blockIdx.x] = data[0];
    }
  }
}

cuda_mnt4_G1 *mnt4_G1_multi_exp_cuda(int len, cuda_mnt4_G1 *d_vec, big_int *d_scalar) {
  int grid_size = (len + BLOCK_SIZE - 1) / BLOCK_SIZE;

  cuda_mnt4_G1 *d_exp, *d_sum;
  cudaMalloc(&d_exp, sizeof(cuda_mnt4_G1) * len);
  cudaMalloc(&d_sum, sizeof(cuda_mnt4_G1) * grid_size);

  printf("len is %d\n", len);
  cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 0xffffffff);

  mnt4_G1_multi_exp_kernel<<<grid_size, BLOCK_SIZE>>>(len, d_exp, d_vec, d_scalar);
  cuda_check_error();

  mnt4_G1_reduce_kernel<<<grid_size, BLOCK_SIZE>>>(len, d_sum, d_vec);
  cuda_check_error();

  cuda_mnt4_G1 *sum = (cuda_mnt4_G1 *) malloc(sizeof(cuda_mnt4_G1) * grid_size);
  cudaMemcpy(sum, d_sum, sizeof(cuda_mnt4_G1) * grid_size, cudaMemcpyDeviceToHost);

  cudaFree(d_exp);
  cudaFree(d_sum);

  cuda_mnt4_G1 res = cuda_mnt4_G1_zero;
  // TODO convert back to normal point format
  /*for (int i = 0; i < grid_size; i++) {
    cuda_mnt4_G1_add(&res, &res, &sum[i]);
  }*/

  return sum;
}

// copying CPU curve points to GPU
cuda_mnt4_G1 *mnt4_G1_to_gpu(
    std::vector<libff::mnt4_G1>::const_iterator vec_start,
    std::vector<libff::mnt4_G1>::const_iterator vec_end) {
  const size_t len = vec_end - vec_start;

  cuda_mnt4_G1 *d_vec;
  cudaError_t err = cudaMalloc(&d_vec, sizeof(cuda_mnt4_G1) * len);
  if (err) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
    std::exit(0);
  }

  // TODO coalesce into a single memcpy?
  for (int i = 0; vec_start + i != vec_end; i++) {
    cuda_mnt4_G1 p;
    for (int l = 0; l < 5; l++) {
      // XXX sorry
      p.X[2*l] = (vec_start + i)->X().mont_repr.data[l] & 0xffffffff;
      p.X[2*l + 1] = (vec_start + i)->X().mont_repr.data[l] >> 32;
      p.Y[2*l] = (vec_start + i)->Y().mont_repr.data[l] & 0xffffffff;
      p.Y[2*l + 1] = (vec_start + i)->Y().mont_repr.data[l] >> 32;
      p.Z[2*l] = (vec_start + i)->Z().mont_repr.data[l] & 0xffffffff;
      p.Z[2*l + 1] = (vec_start + i)->Z().mont_repr.data[l] >> 32;
    }
    cudaMemcpy(&d_vec[i], &p, sizeof(cuda_mnt4_G1), cudaMemcpyHostToDevice);
  }

  /*for (int i = 0; i < 100; i++) {
    printf("%d\n", (vec_start + i)->X().mont_repr.data[0]);
  }*/

  return d_vec;
}

big_int *mnt4_Fr_to_gpu(
    std::vector<libff::mnt4_Fr>::const_iterator scalar_start,
    std::vector<libff::mnt4_Fr>::const_iterator scalar_end) {
  const size_t len = scalar_end - scalar_start;

  big_int *d_scalar;
  cudaError_t err = cudaMalloc(&d_scalar, sizeof(big_int) * len);
  if (err) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
    std::exit(0);
  }

  // TODO coalesce into a single memcpy?
  for (int i = 0; scalar_start + i != scalar_end; i++) {
    big_int v;
    for (int l = 0; l < 5; l++) {
      v[2*l] = (scalar_start + i)->mont_repr.data[l] & 0xffffffff;
      v[2*l + 1] = (scalar_start + i)->mont_repr.data[l] >> 32;
    }
    cudaMemcpy(&d_scalar[i], &v, sizeof(big_int), cudaMemcpyHostToDevice);
  }

  return d_scalar;
}
