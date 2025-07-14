#pragma once

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cub/cub.cuh>
#include <fstream>
#include <iostream>
#include <vector>

#define WARP_SIZE 32
#define MAX_UINT 0xffffffffU
#define SEGSIZE 10000
#define SEGSIZE_FACTOR 80000

static inline void check(cudaError_t err, const char* context) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error: " << context << ": " << cudaGetErrorString(err)
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

#define CHECK(x) check(x, #x)

// return (a + b - 1) / b
static inline uint32_t divup(u_int32_t a, uint32_t b) {
  return (a + b - 1) / b;
}

template <typename T>
std::vector<T> read_file(const char* filename, size_t offset = 0) {
  std::ifstream ifs(filename, std::ios::binary);

  const auto begin = ifs.tellg();
  ifs.seekg(0, std::ios::end);
  const auto end = ifs.tellg();
  const std::size_t len = (end - begin) / sizeof(T);
  ifs.seekg(0);

  std::vector<T> v(len, 0);

  for (std::size_t i = 0; i < len; ++i) {
    ifs.read(reinterpret_cast<char*>(v.data() + i), sizeof(T));
  }

  ifs.close();

  return v;
}

void compare_two_arrays(thrust::device_vector<int>& vec_one,
                        thrust::device_vector<int>& vec_two, int n, int offset,
                        bool sparse_first = 0, bool positions = 0,
                        size_t ones_offset = 0) {
  int* vec_one_ptr = thrust::raw_pointer_cast(vec_one.data());
  int* vec_two_ptr = thrust::raw_pointer_cast(vec_two.data());
  thrust::device_vector<bool> wrongs(n);
  bool* W_ptr = thrust::raw_pointer_cast(wrongs.data());
  auto r = thrust::counting_iterator<int>(0);
  thrust::for_each(r + offset, r + n, [=] __device__(int i) {
    W_ptr[i] = 0;
    if (vec_one_ptr[(size_t)i * (sparse_first ? 2 : 1) +
                    (sparse_first ? positions ? 0 : 1 : 0) + ones_offset] !=
        vec_two_ptr[i]) {
      W_ptr[i] = 1;
      /*       printf("Error: %d != %d, i = %d\n", vec_one_ptr[(size_t)i *
      (sparse_first ? 2 : 1) + (sparse_first ? positions ? 0 : 1 : 0) +
      ones_offset], vec_two_ptr[i], i); */
    } else {
      W_ptr[i] = 0;
      // printf("Correct: %d == %d, i = %d\n", vec_one_ptr[i], vec_two_ptr[i],
      // i);
    }
  });
  int num_wrong = thrust::reduce(wrongs.begin(), wrongs.end(), 0);
  if (num_wrong > 0) {
    std::cout << "!!! There are " << num_wrong
              << " wrong values in the arrays of " << n - offset << std::endl;
  } else {
    std::cout << "All values are correct!" << std::endl;
  }
}

void _compare_two_arrays(thrust::device_vector<int>& vec_one,
                         thrust::device_vector<int>& vec_two, int n,
                         int first_offset = 0) {
  int* vec_one_ptr = thrust::raw_pointer_cast(vec_one.data());
  int* vec_two_ptr = thrust::raw_pointer_cast(vec_two.data());
  thrust::device_vector<bool> wrongs(n);
  bool* W_ptr = thrust::raw_pointer_cast(wrongs.data());
  auto r = thrust::counting_iterator<int>(0);
  thrust::for_each(r, r + n, [=] __device__(int i) {
    W_ptr[i] = 0;
    if (vec_one_ptr[i + first_offset] != vec_two_ptr[i]) {
      if (i % 2) {
        W_ptr[i] = 1;
      }
      if (i % 2 && vec_one_ptr[i + first_offset] > vec_two_ptr[i])
        printf("Error at i: %d: %d > %d\n", i, vec_one_ptr[i + first_offset],
               vec_two_ptr[i]);
      printf("Error at i: %d: %d != %d, pos: %d\n", i,
             vec_one_ptr[i + first_offset], vec_two_ptr[i], vec_two_ptr[i - 1]);
    } /* else {
      W_ptr[i] = 0;
      printf("Correct at i: %d, %d == %d\n", i, vec_one_ptr[i + first_offset],
      vec_two_ptr[i]);
      } */
  });
  int num_wrong = thrust::reduce(wrongs.begin(), wrongs.end(), 0);
  if (num_wrong > 0) {
    std::cout << "!!! There are " << num_wrong
              << " wrong values in the arrays of " << n << std::endl;
  } else {
    std::cout << "All values are correct!" << std::endl;
  }
}

void compare_two_vectors(thrust::device_vector<int32_t>& vec_one,
                         thrust::device_vector<uint32_t>& vec_two, int n) {
  thrust::device_vector<int> d_vec_one(vec_one);
  thrust::device_vector<int> d_vec_two(vec_two);
  compare_two_arrays(d_vec_one, d_vec_two, n, 0);
}