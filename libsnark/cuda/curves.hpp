#include <libff/algebra/curves/mnt/mnt4/mnt4_pp.hpp>
#include <libff/algebra/curves/mnt/mnt6/mnt6_pp.hpp>

#include <libsnark/cuda/big_int.hpp>

struct cuda_mnt4_G1 {
  big_int X;
  big_int Y;
  big_int Z;
};

cuda_mnt4_G1 *mnt4_G1_multi_exp_cuda(int, cuda_mnt4_G1 *, big_int *);
cuda_mnt4_G1 *mnt4_G1_to_gpu(std::vector<libff::mnt4_G1>::const_iterator, std::vector<libff::mnt4_G1>::const_iterator);
big_int *mnt4_Fr_to_gpu(std::vector<libff::mnt4_Fr>::const_iterator, std::vector<libff::mnt4_Fr>::const_iterator);

// TODO
/*
struct cuda_mnt4_G2 {
  // TODO
};

struct cuda_mnt6_G1 {
  // TODO
};

struct cuda_mnt6_G2 {
  // TODO
};

cuda_mnt4_G2 *mnt4_G2_to_gpu(std::vector<mnt4_G2>::const_iterator);
cuda_mnt6_G1 *mnt6_G1_to_gpu(std::vector<mnt6_G1>::const_iterator);
cuda_mnt6_G2 *mnt6_G2_to_gpu(std::vector<mnt6_G2>::const_iterator);

template<typename T1, typename T2>
struct cuda_knowledge_commitment {
  T1 g;
  T2 h;

  cuda_knowledge_commitment<T1,T2>() = default;
  cuda_knowledge_commitment<T1,T2>(const cuda_knowledge_commitment<T1,T2> &other) = default;
  cuda_knowledge_commitment<T1,T2>(const T1 &g, const T2 &h);
};*/

void cuda_mnt4_Fq_mul_test(int n, big_int *r, big_int *a, big_int *b);
void cuda_mnt4_Fq_add_test(int n, big_int *r, big_int *a, big_int *b);
void cuda_mnt4_G1_add_test(int n, cuda_mnt4_G1 *r, cuda_mnt4_G1 *a, cuda_mnt4_G1 *b);
