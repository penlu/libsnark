#include <algorithm>
#include <cassert>
#include <functional>
#include <iostream>
#include <sstream>

#include <libff/algebra/curves/mnt/mnt4/mnt4_pp.hpp>
#include <libff/algebra/curves/mnt/mnt6/mnt6_pp.hpp>
#include <libff/algebra/scalar_multiplication/multiexp.hpp>
#include <libff/common/profiling.hpp>
#include <libff/common/utils.hpp>

#ifdef MULTICORE
#include <omp.h>
#endif

#include <libsnark/knowledge_commitment/kc_multiexp.hpp>
#include <libsnark/reductions/r1cs_to_qap/r1cs_to_qap.hpp>
#include <libff/algebra/scalar_multiplication/multiexp.tcc>

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <libsnark/cuda/curves.hpp>
#include <libsnark/cuda/big_int.hpp>

namespace libff {

template<>
mnt4_G1 multi_exp_inner<mnt4_G1, mnt4_Fr, multi_exp_method_cuda>(
    std::vector<mnt4_G1>::const_iterator vec_start,
    std::vector<mnt4_G1>::const_iterator vec_end,
    std::vector<mnt4_Fr>::const_iterator scalar_start,
    std::vector<mnt4_Fr>::const_iterator scalar_end)
{
    mnt4_G1 result(mnt4_G1::zero());

    size_t len = vec_end - vec_start;

    //cudaProfilerStart();

    // transfer data to GPU
    libff::enter_block("multi_exp data transfer", false);
    cuda_mnt4_G1 *d_vec = mnt4_G1_to_gpu(vec_start, vec_end);
    big_int *d_scalar = mnt4_Fr_to_gpu(scalar_start, scalar_end);
    libff::leave_block("multi_exp data transfer", false);

    // call GPU exponentiation kernel
    libff::enter_block("multi_exp CUDA time!", false);
    cuda_mnt4_G1 *sum = mnt4_G1_multi_exp_cuda(len, d_vec, d_scalar);
    libff::leave_block("multi_exp CUDA time!", false);

    cudaFree(d_vec);
    cudaFree(d_scalar);

    // finish the sum
    libff::enter_block("multi_exp final sum time!", false);
    /*for (int i = 0; i < len; i++) {
      mnt4_Fq X, Y, Z;
      for (int j = 0; j < 5; j++) {
        X.mont_repr.data[j] = (((uint64_t) sum[i].X[2*j + 1]) << 32) | sum[i].X[2*j];
        Y.mont_repr.data[j] = (((uint64_t) sum[i].Y[2*j + 1]) << 32) | sum[i].Y[2*j];
        Z.mont_repr.data[j] = (((uint64_t) sum[i].Z[2*j + 1]) << 32) | sum[i].Z[2*j];
      }

      mnt4_G1 pt(X, Y, Z);
      pt.print_coordinates();
    }*/
    for (int i = 0; i < (len + 128 - 1) / 128; i++) {
      mnt4_Fq X, Y, Z;
      for (int j = 0; j < 5; j++) {
        X.mont_repr.data[j] = (((uint64_t) sum[i].X[2*j + 1]) << 32) | sum[i].X[2*j];
        Y.mont_repr.data[j] = (((uint64_t) sum[i].Y[2*j + 1]) << 32) | sum[i].Y[2*j];
        Z.mont_repr.data[j] = (((uint64_t) sum[i].Z[2*j + 1]) << 32) | sum[i].Z[2*j];
      }

      mnt4_G1 pt(X, Y, Z);
      result = result + pt;
    }
    libff::leave_block("multi_exp final sum time!", false);

    free(sum);

    cudaDeviceSynchronize();
    //cudaProfilerStop();
    cudaDeviceReset();

    //std::cout << "HELLO THERE!!!" << std::endl;
    //std::exit(0);

    return result;
}

template<>
mnt6_G1 multi_exp_inner<mnt6_G1, mnt6_Fr, multi_exp_method_cuda>(
    std::vector<mnt6_G1>::const_iterator vec_start,
    std::vector<mnt6_G1>::const_iterator vec_end,
    std::vector<mnt6_Fr>::const_iterator scalar_start,
    std::vector<mnt6_Fr>::const_iterator scalar_end)
{
    mnt6_G1 result(mnt6_G1::zero());

    typename std::vector<mnt6_G1>::const_iterator vec_it;
    typename std::vector<mnt6_Fr>::const_iterator scalar_it;

    // TODO
    std::cout << "HELLO THERE" << std::endl;
    std::exit(0);

    return result;
}

template<>
libsnark::knowledge_commitment<mnt4_G1, mnt4_G1> multi_exp_inner<libsnark::knowledge_commitment<mnt4_G1, mnt4_G1>, mnt4_Fr, multi_exp_method_cuda>(
    std::vector<libsnark::knowledge_commitment<mnt4_G1, mnt4_G1>>::const_iterator vec_start,
    std::vector<libsnark::knowledge_commitment<mnt4_G1, mnt4_G1>>::const_iterator vec_end,
    std::vector<mnt4_Fr>::const_iterator scalar_start,
    std::vector<mnt4_Fr>::const_iterator scalar_end)
{
    libsnark::knowledge_commitment<mnt4_G1, mnt4_G1> result(libsnark::knowledge_commitment<mnt4_G1, mnt4_G1>::zero());

    typename std::vector<libsnark::knowledge_commitment<mnt4_G1, mnt4_G1>>::const_iterator vec_it;
    typename std::vector<mnt4_Fr>::const_iterator scalar_it;

    // TODO
    std::cout << "HELLO THERE" << std::endl;
    std::exit(0);

    return result;
}

template<>
libsnark::knowledge_commitment<mnt4_G2, mnt4_G1> multi_exp_inner<libsnark::knowledge_commitment<mnt4_G2, mnt4_G1>, mnt4_Fr, multi_exp_method_cuda>(
    std::vector<libsnark::knowledge_commitment<mnt4_G2, mnt4_G1>>::const_iterator vec_start,
    std::vector<libsnark::knowledge_commitment<mnt4_G2, mnt4_G1>>::const_iterator vec_end,
    std::vector<mnt4_Fr>::const_iterator scalar_start,
    std::vector<mnt4_Fr>::const_iterator scalar_end)
{
    libsnark::knowledge_commitment<mnt4_G2, mnt4_G1> result(libsnark::knowledge_commitment<mnt4_G2, mnt4_G1>::zero());

    typename std::vector<libsnark::knowledge_commitment<mnt4_G2, mnt4_G1>>::const_iterator vec_it;
    typename std::vector<mnt4_Fr>::const_iterator scalar_it;

    // TODO
    std::cout << "HELLO THERE" << std::endl;
    std::exit(0);

    return result;
}

template<>
libsnark::knowledge_commitment<mnt6_G1, mnt6_G1> multi_exp_inner<libsnark::knowledge_commitment<mnt6_G1, mnt6_G1>, mnt6_Fr, multi_exp_method_cuda>(
    std::vector<libsnark::knowledge_commitment<mnt6_G1, mnt6_G1>>::const_iterator vec_start,
    std::vector<libsnark::knowledge_commitment<mnt6_G1, mnt6_G1>>::const_iterator vec_end,
    std::vector<mnt6_Fr>::const_iterator scalar_start,
    std::vector<mnt6_Fr>::const_iterator scalar_end)
{
    libsnark::knowledge_commitment<mnt6_G1, mnt6_G1> result(libsnark::knowledge_commitment<mnt6_G1, mnt6_G1>::zero());

    typename std::vector<libsnark::knowledge_commitment<mnt6_G1, mnt6_G1>>::const_iterator vec_it;
    typename std::vector<mnt6_Fr>::const_iterator scalar_it;

    // TODO
    std::cout << "HELLO THERE" << std::endl;
    std::exit(0);

    return result;
}

template<>
libsnark::knowledge_commitment<mnt6_G2, mnt6_G1> multi_exp_inner<libsnark::knowledge_commitment<mnt6_G2, mnt6_G1>, mnt6_Fr, multi_exp_method_cuda>(
    std::vector<libsnark::knowledge_commitment<mnt6_G2, mnt6_G1>>::const_iterator vec_start,
    std::vector<libsnark::knowledge_commitment<mnt6_G2, mnt6_G1>>::const_iterator vec_end,
    std::vector<mnt6_Fr>::const_iterator scalar_start,
    std::vector<mnt6_Fr>::const_iterator scalar_end)
{
    libsnark::knowledge_commitment<mnt6_G2, mnt6_G1> result(libsnark::knowledge_commitment<mnt6_G2, mnt6_G1>::zero());

    typename std::vector<libsnark::knowledge_commitment<mnt6_G2, mnt6_G1>>::const_iterator vec_it;
    typename std::vector<mnt6_Fr>::const_iterator scalar_it;

    // TODO
    std::cout << "HELLO THERE" << std::endl;
    std::exit(0);

    return result;
}

}
