#include <iostream>

#include <libff/algebra/curves/mnt/mnt4/mnt4_pp.hpp>
#include <libff/algebra/fields/fp.hpp>
#include <libsnark/cuda/big_int.hpp>
#include <libsnark/cuda/curves.hpp>

#include <cuda_runtime.h>

const big_int zero = {0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0};
const big_int one = {0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0};
const big_int R = {0x5863845c, 0x18c31a7b, 0xe3b68df5, 0xe9de7a15, 0x28faab40, 0xc5df8587, 0x647b5197, 0x29184098, 0x223d33c3, 0x1c1};
const big_int R2 = {0x5613d220, 0x65acec, 0xbf2bc893, 0xa266a1ad, 0x318850e1, 0x66bd7673, 0xad38d47b, 0x1f32e014, 0xf0918a34, 0x224};
const big_int Rinv = {0x62c6ae8b, 0x5d0dfbf7, 0x8d649ad1, 0x420e2c96, 0x31b3038d, 0x8fbe5fea, 0x5681a41e, 0xc2d37cf3, 0x1b712577, 0x25f};

// test numbers
const big_int a[10] = {
  {0x2265b1f5, 0x91b7584a, 0xd8f16adf, 0xcd613e30, 0xc386bbc4, 0x1027c4d1, 0x414c343c, 0x1e2feb89, 0x7ed4d57b, 0x30b},
  {0xd5f4b3b2, 0x63ca828d, 0x6ec9d286, 0x9b810e76, 0xc324c985, 0xc4647159, 0x8a05a6, 0xb2221a58, 0x7204e52d, 0x110},
  {0x6c0fd4f5, 0xb9d179e0, 0x76f3787, 0x8712b8bc, 0x38c0c8fd, 0xc381e88f, 0x701966a0, 0xf06d3fef, 0x7eed8d14, 0x236},
  {0x5805975, 0x6a8ac4ba, 0xd66b829e, 0xea90a8f0, 0x8e73ca47, 0xec148cb4, 0xa46d6753, 0x19999e3f, 0x2f978d87, 0x284},
  {0x4da98f1d, 0x48beab13, 0x966baea1, 0xf9341c68, 0xe1ea24c4, 0x7fd63116, 0xd8a064df, 0xf0dfb4a5, 0x815a47c5, 0x192},
  {0x2c4a3698, 0x5dfbd3d1, 0x8c7e134f, 0xe1fab9d7, 0xb3fa7aa7, 0xc69d4bd8, 0xacab1a6b, 0xbcfbb050, 0x5fec898f, 0x58},
  {0x7d5c8dfc, 0xbb968a43, 0x7923986, 0x78255d68, 0xb21fbac, 0x4efbc8d6, 0xb410d93c, 0xd92a4aa2, 0xfbb230bb, 0x275},
  {0xc541013d, 0x33138131, 0x8a245e6b, 0xeb8ac8ce, 0xdc3bf364, 0x8c5fe8f8, 0x3b6fe507, 0x678a5aa3, 0x83868a29, 0x160},
  {0xe5e18ba, 0x7b297d0b, 0xdeb8fc4c, 0x5d5f576c, 0x91eb79fa, 0x8ded3c96, 0x3328ad08, 0xf0e642f4, 0x81355c53, 0x1a7},
  {0x9cc9af4e, 0x54c56c9a, 0x75491bc3, 0x99901c04, 0x7295e42, 0xcdf84404, 0x3ac7652c, 0xa2a7ae1f, 0x2d5db79b, 0x233},
};

const big_int b[10] = {
  {0x7311d8a3, 0x78e51061, 0xa6cecc1b, 0x612e7696, 0xc9e9c616, 0x35bf992d, 0x18072e8c, 0x7ce42c82, 0x741c7a8, 0x392},
  {0xb8b6d8fe, 0xcd447e35, 0x3a902931, 0x9755d4c1, 0xf1fd42a2, 0x1a2b8f1f, 0xe6c3f339, 0x51431193, 0x7d4bedc, 0x16},
  {0x3bab6c39, 0x587fd280, 0x3b1a11df, 0xad45f23d, 0x380208a9, 0xc2cd789a, 0x75a89294, 0xf3c64af7, 0x4a2f20aa, 0x3b4},
  {0xb610a9f7, 0x803468b6, 0xefba91fc, 0xf79b17ae, 0x6c0f3459, 0x81f9c1f6, 0xd47d380d, 0xe901e35c, 0xab99254a, 0xc2},
  {0x96c8da19, 0xda711448, 0x8d6af57, 0x7af027bc, 0x3e2434e3, 0xbe6521cc, 0xcc22af58, 0x677f6cbd, 0x6a107b75, 0x2a8},
  {0x705fca16, 0xa9ec0806, 0x82283d15, 0x1ba16215, 0xc74803e3, 0x29e821a4, 0x855c3844, 0xd707107e, 0x64ac5db9, 0x17b},
  {0x97dae38d, 0x9403560d, 0x64c2f2e3, 0xa5ac06d8, 0x2b9c014e, 0x2b28fef0, 0x8092b4d4, 0x3a1890c7, 0xfb695ffb, 0xc},
  {0xf3d4e711, 0xd8f33418, 0x93ea5c4e, 0x5a702cfa, 0x7589a82b, 0xe8e5b461, 0x44ef7feb, 0xa8c24d42, 0x8c497c68, 0x26f},
  {0x7c240d49, 0xd037cdff, 0x5b569643, 0x6a17b9af, 0x58989008, 0x67dba8, 0x89d9bf02, 0x8a449ebe, 0x9f9d0129, 0x325},
  {0x959f3a51, 0x2e47dc0e, 0xdc6b13ab, 0x1773308c, 0xcc667e97, 0x8d103ed3, 0xcc0e95ee, 0xd9ed17e3, 0xd1020a15, 0x3b9},
};

// c[i] should be a[i] * b[i]
const big_int c[10] = {
  {0xac884a7d, 0x1954b54c, 0x5a45d6e7, 0xe87748b5, 0xfc73758f, 0xbde89213, 0xb6727d99, 0xb76882ee, 0x8d3f85f0, 0x18e},
  {0x52a04326, 0xdf095e36, 0x50030c85, 0x6baad027, 0x4ada7227, 0x5079061f, 0xd70da800, 0x341f594d, 0x77532493, 0x1a6},
  {0x91c0f457, 0x112244d7, 0x33cedb42, 0x6c53e78, 0xa79d93f3, 0xe6d5291e, 0x30ce2d8b, 0x15614af0, 0x6ae39f5, 0x1bc},
  {0xae18b26d, 0xdda3e0fd, 0x25a4282f, 0x4258cf73, 0x865ed43, 0x2c28d4b7, 0x80c9e9ef, 0xd112463d, 0x527b940e, 0xe9},
  {0x180cfae2, 0x28ec435f, 0xcd250468, 0xe3b7637d, 0x6fcce98, 0x8dd8cd88, 0xbd8f6b35, 0x892c6fef, 0xa80bb783, 0x1b6},
  {0xbfc786d7, 0x9b2b8572, 0xc7c08b9d, 0x9ea51fe5, 0x2a4f8e01, 0x164d68e2, 0x1f4f35c9, 0x1d8973cf, 0x2a92259d, 0x15},
  {0xbf802228, 0x7fdf3656, 0x3e9442c4, 0xceeeaccf, 0x84f8e68e, 0x2120a6b9, 0x5e64f6ac, 0xb0f22b8a, 0x87b117c3, 0x83},
  {0x92488bf5, 0x798afeb9, 0x319d4c2e, 0x73afbf60, 0x1e18adea, 0xa5694b68, 0xa68db91e, 0xdf85e21e, 0x43478db0, 0x268},
  {0xa1aaab1b, 0x99418f94, 0x531b6a35, 0xbfa6350, 0xa352a8f7, 0xaffc7796, 0xe705a037, 0xe3212ee5, 0xf5f04b22, 0x2e6},
  {0xb9fdf984, 0x82a6fc63, 0x8285cb43, 0x4015f029, 0x272d8826, 0x53076961, 0x5919ab91, 0x5e52d9f6, 0x21c1c873, 0x341},
};

// d[i] should be a[i] + b[i]
const big_int d[10] = {
  {0x24118a97, 0x418f9251, 0x2ea028e8, 0xece5d169, 0x305d50f0, 0x7af8949c, 0xb1a30e39, 0xf8adf36d, 0x8e59c8af, 0x2e0},
  {0x8eab8cb0, 0x310f00c3, 0xa959fbb8, 0x32d6e337, 0xb5220c28, 0xde900079, 0xe74df8df, 0x3652beb, 0x79d9a40a, 0x126},
  {0x3655412d, 0x49447606, 0xf1693b54, 0xf2aec79a, 0x13afa0bc, 0xbb6097c6, 0x3e11a4a6, 0x41cd6649, 0xd15fd94c, 0x22d},
  {0xbb91036c, 0xeabf2d70, 0xc626149a, 0xe22bc09f, 0xfa82fea1, 0x6e0e4eaa, 0x78ea9f61, 0x29b819c, 0xdb30b2d2, 0x346},
  {0x730c6935, 0x5a22e901, 0x4e224fe6, 0x327a60c6, 0xc2fb28be, 0x734c897f, 0xfd12bfa9, 0xb5f8fcc5, 0xf3adeec7, 0x7d},
  {0x9caa00ae, 0x7e7dbd7, 0xea65065, 0xfd9c1bed, 0x7b427e8a, 0xf0856d7d, 0x320752af, 0x9402c0cf, 0xc498e749, 0x1d3},
  {0x15377189, 0x4f99e051, 0x6c552c6a, 0x1dd16440, 0x36bdfcfb, 0x7a24c7c6, 0x34a38e10, 0x1342db6a, 0xf71b90b7, 0x282},
  {0x47afe84d, 0x42f9def0, 0xcceeaca7, 0x451126a, 0xf4b26aa6, 0xaa56d3f6, 0xd8af1064, 0x6de68347, 0x1813321e, 0x13},
  {0x191c2602, 0x825474b0, 0xe8ef847d, 0x85cd2dbd, 0x8d70d918, 0xc3664edb, 0x1552177b, 0xd8c4bd15, 0x29158909, 0x110},
  {0xc102e99e, 0xba00724e, 0x94215b, 0x6f596933, 0x767cabef, 0x9019b974, 0x5f25a68c, 0xda2ea165, 0x6a2ed3d, 0x230},
};

libff::mnt4_Fq big_int_to_mnt4_Fq(const big_int x) {
  libff::mnt4_Fq X;
  for (int i = 0; i < 5; i++) {
    X.mont_repr.data[i] = ((uint64_t) x[2*i + 1]) << 32 | x[2*i];
  }
  return X;
}

void mnt4_Fq_to_big_int(const libff::mnt4_Fq X, big_int x) {
  for (int i = 0; i < 5; i++) {
    x[2*i] = X.mont_repr.data[i] & 0xffffffff;
    x[2*i + 1] = X.mont_repr.data[i] >> 32;
  }
}

int main() {
  std::cout << "Hello world!" << std::endl;

  libff::mnt4_pp::init_public_params();

  // sanity check: CPU multiplication
  libff::mnt4_Fq SA[10];
  libff::mnt4_Fq SB[10];
  libff::mnt4_Fq SC[10];
  libff::mnt4_Fq SD[10];
  for (int i = 0; i < 10; i++) {
    SA[i] = big_int_to_mnt4_Fq(a[i]);
    SB[i] = big_int_to_mnt4_Fq(b[i]);
    SC[i] = big_int_to_mnt4_Fq(c[i]);
    SD[i] = big_int_to_mnt4_Fq(d[i]);
    if (SA[i] * SB[i] != SC[i]) {
      std::cout << "cpu mul is bad" << std::endl;
      std::exit(0);
    }
    if (SA[i] + SB[i] != SD[i]) {
      std::cout << "cpu add is bad" << std::endl;
      std::exit(0);
    }
  }

  // GPU tests
  big_int *d_r, *d_a, *d_b;
  big_int r[10];
  cudaMalloc(&d_r, 10 * sizeof(big_int));
  cudaMalloc(&d_a, 10 * sizeof(big_int));
  cudaMalloc(&d_b, 10 * sizeof(big_int));

  // test addition
  cudaMemcpy(d_a, a, 10 * sizeof(big_int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, 10 * sizeof(big_int), cudaMemcpyHostToDevice);
  cuda_mnt4_Fq_add_test(10, d_r, d_a, d_b);
  cudaMemcpy(r, d_r, 10 * sizeof(big_int), cudaMemcpyDeviceToHost);

  for (int i = 0; i < 10; i++) {
    libff::mnt4_Fq R = big_int_to_mnt4_Fq(r[i]);
    if (SD[i] != R) {
      std::cout << "gpu add is bad" << std::endl;
      std::exit(0);
    }
  }

  // test multiplication
  cuda_mnt4_Fq_mul_test(10, d_r, d_a, d_b);
  cudaMemcpy(r, d_r, 10 * sizeof(big_int), cudaMemcpyDeviceToHost);
  for (int i = 0; i < 10; i++) {
    libff::mnt4_Fq R = big_int_to_mnt4_Fq(r[i]);
    if (SC[i] != R) {
      std::cout << "gpu mul is bad" << std::endl;
      std::exit(0);
    }
  }

  // prepare curve points by exponentiation
  libff::mnt4_G1 CA[10];
  libff::mnt4_G1 CB[10];
  for (int i = 0; i < 10; i++) {
    CA[i] = SA[i] * libff::mnt4_G1::G1_one;
    CB[i] = SB[i] * libff::mnt4_G1::G1_one;
  }

  // copy points to GPU
  cuda_mnt4_G1 *d_CA, *d_CB, *d_CR;
  cudaMalloc(&d_CA, sizeof(cuda_mnt4_G1) * 10);
  cudaMalloc(&d_CB, sizeof(cuda_mnt4_G1) * 10);
  cudaMalloc(&d_CR, sizeof(cuda_mnt4_G1) * 10);
  for (int i = 0; i < 10; i++) {
    cuda_mnt4_G1 p;
    mnt4_Fq_to_big_int(CA[i].X(), p.X);
    mnt4_Fq_to_big_int(CA[i].Y(), p.Y);
    mnt4_Fq_to_big_int(CA[i].Z(), p.Z);
    cudaMemcpy(&d_CA[i], &p, sizeof(cuda_mnt4_G1), cudaMemcpyHostToDevice);
  }
  for (int i = 0; i < 10; i++) {
    cuda_mnt4_G1 p;
    mnt4_Fq_to_big_int(CB[i].X(), p.X);
    mnt4_Fq_to_big_int(CB[i].Y(), p.Y);
    mnt4_Fq_to_big_int(CB[i].Z(), p.Z);
    cudaMemcpy(&d_CB[i], &p, sizeof(cuda_mnt4_G1), cudaMemcpyHostToDevice);
  }

  // GPU point addition
  cuda_mnt4_G1 CR[10];
  cuda_mnt4_G1_add_test(10, d_CR, d_CA, d_CB);
  cudaMemcpy(CR, d_CR, 10 * sizeof(cuda_mnt4_G1), cudaMemcpyDeviceToHost);
  for (int i = 0; i < 10; i++) {
    libff::mnt4_Fq X, Y, Z;
    X = big_int_to_mnt4_Fq(CR[i].X);
    Y = big_int_to_mnt4_Fq(CR[i].Y);
    Z = big_int_to_mnt4_Fq(CR[i].Z);
    libff::mnt4_G1 cr(X, Y, Z);
    libff::mnt4_G1 correct = CA[i] + CB[i];
    if (cr != CA[i] + CB[i]) {
      std::cout << "test " << i << std::endl;
      cr.X().mont_repr.print_hex();
      cr.Y().mont_repr.print_hex();
      cr.Z().mont_repr.print_hex();
      correct.X().mont_repr.print_hex();
      correct.Y().mont_repr.print_hex();
      correct.Z().mont_repr.print_hex();
      std::cout << "gpu point add is bad" << std::endl;
      std::exit(0);
    }
  }

  return 0;
}
