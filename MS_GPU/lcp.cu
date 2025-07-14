#include <chrono>
#include <cstdlib>
#include <iostream>

#include "lcp_utils.cu"
#include "libcu_utils.cu"
#include "utils.cu"
/*
void test_LCP_segment_size(thrust::device_vector<uint8_t>& T,
                           thrust::device_vector<int32_t>& sa,
                           thrust::device_vector<int32_t>& phi,
                           thrust::device_vector<int32_t>& isa,
                           thrust::device_vector<int32_t>& lcp, size_t n) {
  using std::chrono::duration;
  using std::chrono::duration_cast;
  using std::chrono::high_resolution_clock;
  using std::chrono::milliseconds;

  auto start = high_resolution_clock::now();
  compute_phi(sa, phi, n);
  CHECK(cudaGetLastError());
  cudaDeviceSynchronize();
  auto end = high_resolution_clock::now();
  duration<double, std::milli> ms_double = end - start;

  start = high_resolution_clock::now();
  compute_ISA(sa, isa, n);
  CHECK(cudaGetLastError());
  cudaDeviceSynchronize();
  end = high_resolution_clock::now();
  ms_double = end - start;

  int32_t block_size = WARP_SIZE;
  for (int b = 10000; b <= 100000000; b *= 2) {
    int segment_size = divup(n, b);
    int32_t bl_num = divup(n, block_size * segment_size);
    std::cout << "\n\tTesting LCP with segment size: " << segment_size
              << "\tb: " << b << " blocksize " << block_size << " blocknum "
              << bl_num << std::endl;
    start = high_resolution_clock::now();
    compute_phi(sa, phi, n);
    CHECK(cudaGetLastError());
    cudaDeviceSynchronize();
    end = high_resolution_clock::now();
    ms_double = end - start;
    std::cout << "\n===>\tComputing phi on GPU of file size: " << n
              << " took: " << ms_double.count() << " \n"
              << std::endl;

    start = high_resolution_clock::now();
    compute_PLCP(T, phi, n, block_size, bl_num, segment_size);
    CHECK(cudaGetLastError());
    cudaDeviceSynchronize();
    end = high_resolution_clock::now();
    ms_double = end - start;
    std::cout << "\n1==>\tComputing PLCP on GPU with segment size: "
              << segment_size << " took: \t\t" << ms_double.count()
              << std::endl;

    start = high_resolution_clock::now();
    unpermute_PLCP(sa, phi, lcp, n);
    CHECK(cudaGetLastError());
    cudaDeviceSynchronize();
    end = high_resolution_clock::now();
    ms_double = end - start;
    std::cout << "\n===>\tUnpermute PLCP on GPU of file size: " << n
              << " took: " << ms_double.count() << " \n"
              << std::endl;

    start = high_resolution_clock::now();
    compute_ISA(sa, isa, n);
    CHECK(cudaGetLastError());
    cudaDeviceSynchronize();
    end = high_resolution_clock::now();
    ms_double = end - start;
    std::cout << "\n===>\tComputing ISA on GPU of file size: " << n
              << " took: " << ms_double.count() << " \n"
              << std::endl;
    start = high_resolution_clock::now();
    compute_kasai_LCP(T, sa, isa, lcp, n, block_size, bl_num, segment_size);
    CHECK(cudaGetLastError());
    cudaDeviceSynchronize();
    end = high_resolution_clock::now();
    ms_double = end - start;
    std::cout << "\n2==>\tComputing Kasai LCP on GPU with segment size: "
              << segment_size << " took: \t" << ms_double.count() << std::endl;
  }
}
  */

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s [input text file]\n", argv[0]);
    std::exit(1);
  }

  using std::chrono::duration;
  using std::chrono::duration_cast;
  using std::chrono::high_resolution_clock;
  using std::chrono::milliseconds;

  auto input_text = read_file<uint8_t>(argv[1]);
  size_t n = input_text.size();
  if (n > INT32_MAX) {
    std::fprintf(stderr, "Input file size too big for SA construction");
    std::exit(1);
  }

  std::cout << "\nLCP construction testing, file: " << argv[1]
            << ", size: " << n << " bytes\n"
            << std::endl;

  auto start = high_resolution_clock::now();
  auto end = high_resolution_clock::now();
  duration<double, std::milli> ms_double = end - start;

  thrust::device_vector<uint8_t> T(input_text);
  thrust::device_vector<uint> SA(n);
  CHECK(cudaGetLastError());
  cudaDeviceSynchronize();
  std::cout << "\nMem alloc and copy to device GPU took: " << ms_double.count()
            << " milliseconds\n"
            << std::endl;

  Compute_SA_libcu(T, SA, n, 1);
  CHECK(cudaGetLastError());
  cudaDeviceSynchronize();

  thrust::device_vector<int32_t> phi(n);
  thrust::device_vector<int32_t> isa(n);
  thrust::device_vector<int32_t> lcp(n);
  compute_LCP_standard(T, SA, lcp, n, 0);
  // test_LCP_segment_size(T, SA, phi, isa, lcp, n);
}