#include <chrono>
#include <cstdlib>
#include <iostream>

#include "lcp_utils.cu"
#include "libcu_utils.cu"
#include "merge_utils.cu"
#include "ms_utils.cu"
#include "utils.cu"

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
  size_t i = 0, reference_sz;
  while (true) {
    if (input_text[i] == '$') {
      reference_sz = i + 1;
      input_text[i] = (char)0;
      break;
    }
    ++i;
  }

  std::cout << "\nMatching Statistics construction by SA merge, file: "
            << argv[1] << ", size: " << n << " bytes\n"
            << std::endl;
  int chunk_factor = SEGSIZE_FACTOR;
  auto start = high_resolution_clock::now();

  /* thrust::device_vector<uint8_t> T(input_text); */

  thrust::device_vector<uint8_t> R(input_text.begin(),
                                   input_text.begin() + reference_sz);
  thrust::device_vector<uint8_t> S(input_text.begin() + reference_sz,
                                   input_text.end());
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  auto end = high_resolution_clock::now();
  duration<double, std::milli> ms_double = end - start;
  std::cout << "\nMem alloc and copy to device GPU took: " << ms_double.count()
            << " milliseconds\n"
            << std::endl;

  // Computing M_l sequentially via SA_R, SA_S and LCP_R, LCP_S
  /* std::vector<uint8_t> R_host(input_text.begin(),
                              input_text.begin() + reference_sz); */
  thrust::device_vector<uint> SA_R(reference_sz);
  auto sa_time = Compute_SA_libcu(R, SA_R, reference_sz, 1);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  thrust::device_vector<uint> SA_S(n - reference_sz);
  /* std::vector<uint8_t> S_host(input_text.begin() + reference_sz,
                              input_text.end()); */
  auto total_time = sa_time;
  sa_time = Compute_SA_libcu(S, SA_S, n - reference_sz, 1);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  total_time += sa_time;
  thrust::device_vector<int> LCP_R(reference_sz);
  thrust::device_vector<int> LCP_S(n - reference_sz);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  total_time += compute_LCP_standard(R, SA_R, LCP_R, reference_sz);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  total_time += compute_LCP_standard(S, SA_S, LCP_S, n - reference_sz);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  int chunk_size = divup(n - reference_sz, chunk_factor);
  int num_chunks = divup(n - reference_sz, chunk_size);
  thrust::device_vector<int> offsets_and_lens(num_chunks * 2);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  std::cout << "\nChunk size: " << chunk_size << ", num_chunks: " << num_chunks
            << ", chunk_factor: " << chunk_factor << std::endl;
  start = high_resolution_clock::now();
  get_matching_factor_offsets(SA_R, SA_S, R, S, offsets_and_lens, reference_sz,
                              n - reference_sz, chunk_size);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  end = high_resolution_clock::now();
  ms_double = end - start;
  std::cout << "\n===> Get offsets and lens took: " << ms_double.count()
            << " milliseconds\n"
            << std::endl;

  total_time += ms_double.count();
  // compute the Matching Statistics directly from the SA and the LCP arrays
  thrust::device_vector<int> MS((n - reference_sz) * 2);

  start = high_resolution_clock::now();
  get_factors_directly_from_factors(R, S, SA_R, SA_S, LCP_R, LCP_S,
                                    offsets_and_lens, MS, chunk_size);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  end = high_resolution_clock::now();
  ms_double = end - start;
  std::cout << "\n===> Compute MS directly from SA and LCP took: "
            << ms_double.count() << " milliseconds\n"
            << std::endl;
  total_time += ms_double.count();
  std::cout << "\n===> Total time for MS construction: " << total_time
            << " milliseconds\n"
            << std::endl;
  std::vector<int> MS_cpu(MS.size());
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  start = high_resolution_clock::now();
  thrust::copy(MS.begin(), MS.end(), MS_cpu.begin());
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  end = high_resolution_clock::now();
  ms_double = end - start;
  std::cout << "\n===> Copying MS to host took: " << ms_double.count()
            << " milliseconds\n"
            << std::endl;

  thrust::device_vector<int> flags(S.size(), 0);
  start = high_resolution_clock::now();
  mark_CMS_heads_from_MS(MS, flags, S.size());
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  end = high_resolution_clock::now();
  ms_double = end - start;
  std::cout << "\n===> Marking CMS heads from MS took: " << ms_double.count()
            << " milliseconds\n"
            << std::endl;
  thrust::device_vector<int> ranks(S.size(), 0);
  start = high_resolution_clock::now();
  UpdateRanks(flags, ranks, S.size());
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  end = high_resolution_clock::now();
  ms_double = end - start;
  std::cout << "\n===> Updating ranks from flags took: " << ms_double.count()
            << " milliseconds\n"
            << std::endl;

  start = high_resolution_clock::now();
  thrust::device_vector<int>::iterator it = ranks.end() - 1;
  int num_heads = *it;
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  end = high_resolution_clock::now();
  ms_double = end - start;
  std::cout << "\n#heads: " << num_heads << " out of " << n - reference_sz
            << ", (" << (float)num_heads / (n - reference_sz)
            << ") values, retrieving this number took " << ms_double.count()
            << " milliseconds\n"
            << std::endl;
  thrust::device_vector<int> CMS(num_heads * 3);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  start = high_resolution_clock::now();
  write_CMS_values_from_MS(CMS, flags, ranks, MS, S.size());
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  end = high_resolution_clock::now();
  ms_double = end - start;
  std::cout << "\nreference size " << R.size() << ", S size: " << S.size()

            << std::endl;
  std::cout << "\n===> Write CMS values from MS took: " << ms_double.count()
            << " milliseconds\n"
            << std::endl;

  std::vector<int> CMS_cpu(CMS.size());
  start = high_resolution_clock::now();
  thrust::copy(CMS.begin(), CMS.end(), CMS_cpu.begin());
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  end = high_resolution_clock::now();
  ms_double = end - start;
  std::cout << "\n===> Copying CMS to host took: " << ms_double.count()
            << " milliseconds\n"
            << std::endl;
}