#include <thrust/functional.h>
#include <thrust/reduce.h>

#include <chrono>
#include <cstdlib>
#include <iostream>

#include "lcp_utils.cu"
#include "libcu_utils.cu"
#include "lmp_utils.cu"
#include "utils.cu"

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s [input text file] \n", argv[0]);
    std::exit(1);
  }

  using std::chrono::duration;
  using std::chrono::duration_cast;
  using std::chrono::high_resolution_clock;
  using std::chrono::milliseconds;

  auto input_text = read_file<int8_t>(argv[1]);
  /* bool lrf_heuristic = 1;
  if (argc > 2) {
    lrf_heuristic = (bool)argv[2];
  } */
  size_t n = input_text.size();
  if (n > INT32_MAX) {
    std::fprintf(stderr, "Input file size too big for SA construction");
    std::exit(1);
  }

  std::vector<uint32_t> offs;
  size_t i = 0, reference_sz = 0, max_sz;
  bool first = 1;
  while (i < n) {
    if (input_text[i] == '$') {
      if (first) {
        reference_sz = i + 1;
        input_text[i] = (char)0;
        first = 0;
        continue;
      }
      offs.push_back(i - reference_sz);
      if (offs[offs.size() - 1] - offs[offs.size() - 2] > max_sz)
        max_sz = offs[offs.size() - 1] - offs[offs.size() - 2];
      // std::cout << i - reference_sz << "\n";
    }
    ++i;
  }

  std::cout << "\nPrefix doubling SA construction (GPU implementation), file: "
            << argv[1] << ", size: " << n << " bytes\n"
            << "Reference size: " << reference_sz
            << "\tNumber of Docs: " << offs.size()
            << "\tlargest Doc size: " << max_sz << std::endl;

  auto start = high_resolution_clock::now();

  thrust::device_vector<uint8_t> Reference(input_text.begin(),
                                          input_text.begin() + reference_sz);
  thrust::device_vector<uint8_t> Collection(input_text.begin() + reference_sz,
                                           input_text.begin() + n);
  thrust::device_vector<uint> SA_R(reference_sz);
  thrust::device_vector<uint> ISA_R(reference_sz);
  auto end = high_resolution_clock::now();
  duration<double, std::milli> ms_double = end - start;

  std::cout << "\nMem alloc and copy to device (thrust) took: "
            << ms_double.count() << " milliseconds\n"
            << std::endl;

  auto sa_time = Compute_SA_libcu(Reference, SA_R, reference_sz, 0);

  CHECK(cudaGetLastError());
  cudaDeviceSynchronize();

  std::cout << "\nPrefix doubling SA_R construction on GPU (excluding memory "
               "transfers to/from device) took: "
            << sa_time << " milliseconds\n"
            << std::endl;
  auto total_time = sa_time;
  thrust::device_vector<uint32_t> Offs(offs);

  thrust::device_vector<int> LCP_R(reference_sz);
  thrust::device_vector<int> LRF_R(reference_sz);

  CHECK(cudaGetLastError());
  cudaDeviceSynchronize();

  start = high_resolution_clock::now();
  compute_ISA(SA_R, ISA_R, reference_sz);
  CHECK(cudaGetLastError());
  cudaDeviceSynchronize();

  end = high_resolution_clock::now();
  ms_double = end - start;
  std::cout << "\ncompute_ISA took: " << ms_double.count() << " milliseconds\n"
            << std::endl;

  total_time += ms_double.count();
  auto lcp_time = compute_LCP_standard(Reference, SA_R, LCP_R, reference_sz, 0);
  CHECK(cudaGetLastError());
  cudaDeviceSynchronize();

  std::cout << "\ncompute_LCP took: " << lcp_time << " milliseconds\n"
            << std::endl;
  total_time += lcp_time;

  start = high_resolution_clock::now();
  write_LRF(SA_R, LRF_R, LCP_R, reference_sz);
  CHECK(cudaGetLastError());
  cudaDeviceSynchronize();

  end = high_resolution_clock::now();
  ms_double = end - start;
  std::cout << "\nwrite_LRF took: " << ms_double.count() << " milliseconds\n"
            << std::endl;
  total_time += ms_double.count();
  start = high_resolution_clock::now();

  int result = thrust::reduce(LCP_R.data(), LCP_R.data() + reference_sz, -1,
                              thrust::maximum<int>());
  int lcp_max = result;

  end = high_resolution_clock::now();
  ms_double = end - start;
  std::cout << "\nfinding max_LCP value (" << lcp_max
            << ") took: " << ms_double.count() << " milliseconds\n"
            << std::endl;
  total_time += ms_double.count();
  thrust::device_vector<int32_t> MS((n - reference_sz) * 2);
  std::cout << "\nTotal time for SA, ISA, LCP and LRF construction: "
            << total_time << " milliseconds\n"
            << std::endl;

  uint8_t* col_ptr = thrust::raw_pointer_cast(Collection.data());
  uint32_t* offs_ptr = thrust::raw_pointer_cast(Offs.data());
  uint8_t* ref_ptr = thrust::raw_pointer_cast(Reference.data());
  uint32_t* sa_ptr = thrust::raw_pointer_cast(SA_R.data());
  uint32_t* isa_ptr = thrust::raw_pointer_cast(ISA_R.data());
  int32_t* lcp_ptr = thrust::raw_pointer_cast(LCP_R.data());
  int32_t* lrf_ptr = thrust::raw_pointer_cast(LRF_R.data());
  int32_t* ms_ptr = thrust::raw_pointer_cast(MS.data());
  double best[3] = {UINT32_MAX, UINT32_MAX, UINT32_MAX};

  for (size_t seg_sz = 32; seg_sz >= 1; seg_sz /= 2) {
    int bl_sz = WARP_SIZE * 32;
    size_t blocks = divup(max_sz, seg_sz * bl_sz);
    thrust::device_vector<uint32_t> stat_vec(Offs.size() * blocks);
    {
      uint32_t* stats_ptr = thrust::raw_pointer_cast(stat_vec.data());
      CHECK(cudaGetLastError());
      cudaDeviceSynchronize();
      std::cout << "\n\n>MS construction on GPU with " << blocks
                << " blocks per string and blocksize: " << bl_sz
                << " segment size per thread: " << seg_sz << "\n"
                << std::endl;
      start = high_resolution_clock::now();

      compute_MS<<<Offs.size() * blocks, bl_sz>>>(
          col_ptr, offs_ptr, ref_ptr, sa_ptr, isa_ptr, lcp_ptr, lrf_ptr, ms_ptr,
          reference_sz, 1, blocks, seg_sz);
      CHECK(cudaGetLastError());
      cudaDeviceSynchronize();

      end = high_resolution_clock::now();
      ms_double = end - start;
      std::cout << "MS construction on GPU (excluding memory "
                   "transfers to/from device) with lrf_heuristic took: "
                << ms_double.count() << " milliseconds\n"
                << std::endl;
      if (ms_double.count() < best[0]) {
        best[0] = ms_double.count();
      }
      start = high_resolution_clock::now();

      compute_MS<<<Offs.size() * blocks, bl_sz>>>(
          col_ptr, offs_ptr, ref_ptr, sa_ptr, isa_ptr, lcp_ptr, lrf_ptr, ms_ptr,
          reference_sz, 0, blocks, seg_sz);
      CHECK(cudaGetLastError());
      cudaDeviceSynchronize();

      end = high_resolution_clock::now();
      ms_double = end - start;
      std::cout << "MS construction on GPU (excluding memory "
                   "transfers to/from device) without lrf_heuristic took: "
                << ms_double.count() << " milliseconds\n"
                << std::endl;
      if (ms_double.count() < best[0]) {
        best[1] = ms_double.count();
      }
      start = high_resolution_clock::now();

      compute_MS<<<Offs.size() * blocks, bl_sz>>>(
          col_ptr, offs_ptr, ref_ptr, sa_ptr, isa_ptr, lcp_ptr, lrf_ptr, ms_ptr,
          reference_sz, 0, blocks, seg_sz, lcp_max);
      CHECK(cudaGetLastError());
      cudaDeviceSynchronize();

      end = high_resolution_clock::now();
      ms_double = end - start;
      std::cout << "MS construction on GPU (excluding memory "
                   "transfers to/from device) without lrf_heuristic, using "
                   "max_lcp value took: "
                << ms_double.count() << " milliseconds\n"
                << std::endl;

      if (ms_double.count() < best[0]) {
        best[2] = ms_double.count();
      }
    }
    CHECK(cudaGetLastError());
    cudaDeviceSynchronize();
  }

  std::vector<int32_t> MS_h((n - reference_sz) * 2);

  thrust::copy(MS.begin(), MS.end(), MS_h.begin());

  CHECK(cudaGetLastError());
  cudaDeviceSynchronize();

  // write .ms to file in binary

  /* auto out_file = std::string(argv[1]) + ".GPU.lmp_ms";
  std::ofstream ofs(out_file, std::ios::binary);
  ofs.write(reinterpret_cast<char*>(MS_h.data()),
            sizeof(decltype(MS_h)::value_type) * MS_h.size());
  ofs.close(); */
}