#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/transform.h>

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
    std::fprintf(stderr,
                 "usage: %s [input text file]"
                 "default = 1] \n",
                 argv[0]);
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

  std::vector<uint32_t> offs;
  size_t i = 0, reference_sz = 0;
  bool first = 1;
  while (i < n) {
    if (input_text[i] == '$') {
      if (first) {
        reference_sz = i + 1;
        input_text[i] = (char)0;
        first = 0;
      }
      offs.push_back(i + 1);
    }
    ++i;
  }
  int chunk_factor = SEGSIZE_FACTOR;
  std::cout << "\nMatching Statistics construction by SA merge, file: "
            << argv[1] << ", size: " << n << " bytes\n"
            << std::endl;

  auto start = high_resolution_clock::now();

  thrust::device_vector<uint8_t> T(input_text);

  auto end = high_resolution_clock::now();
  duration<double, std::milli> ms_double = end - start;

  std::cout << "\nMem alloc and copy to device GPU took: " << ms_double.count()
            << " milliseconds\n"
            << std::endl;

  thrust::device_vector<uint> SA_R(reference_sz);
  start = high_resolution_clock::now();
  thrust::device_vector<uint8_t> R(T.begin(), T.begin() + reference_sz);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  end = high_resolution_clock::now();
  ms_double = end - start;
  auto gpu_copy_time = ms_double.count();
  auto sa_time = Compute_SA_libcu(R, SA_R, reference_sz, 0);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  thrust::device_vector<int> LCP_R(reference_sz);
  auto lcp_time_first = compute_LCP_standard(R, SA_R, LCP_R, reference_sz, 0);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  std::vector<double> times(4, 0);  // 0: SA, 2: LCP, 1: offsets, 3: MS
  times[0] = sa_time;
  double copy_time = 0;
  auto sa_time_first = sa_time;
  // int files_at_time = 1;
  std::vector<double> best_times(6, 0);  // 0: SA, 2: LCP,
  best_times[4] = (double)INT_MAX;

  thrust::device_vector<uint> SA_S_full(n - reference_sz);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  int chunk_sz;
  for (int files_at_time = 1; files_at_time <= 128 * 8; files_at_time *= 2) {
    if (files_at_time > offs.size() - 1) {
      break;
    }
    copy_time = 0;
    times[0] = sa_time_first;
    times[1] = lcp_time_first;
    for (int s_file = 0; s_file < offs.size() - files_at_time;
         s_file += files_at_time) {
      auto s_start = offs[s_file];
      auto s_end = offs[min(s_file + files_at_time, (int)offs.size() - 1)];

      start = high_resolution_clock::now();
      thrust::device_vector<uint8_t> S(T.begin() + s_start, T.begin() + s_end);
      CHECK(cudaGetLastError());
      CHECK(cudaDeviceSynchronize());
      end = high_resolution_clock::now();
      ms_double = end - start;
      copy_time += ms_double.count();
      int s_n = s_end - s_start;
      thrust::device_vector<uint> SA_S(s_n);
      CHECK(cudaGetLastError());
      CHECK(cudaDeviceSynchronize());
      sa_time = Compute_SA_libcu(S, SA_S, s_n, 0);
      times[0] += sa_time;
      CHECK(cudaGetLastError());
      CHECK(cudaDeviceSynchronize());

      thrust::device_vector<int> LCP_S(s_n);
      auto lcp_time = compute_LCP_standard(S, SA_S, LCP_S, s_n, 0);
      times[1] += lcp_time;
      CHECK(cudaGetLastError());
      CHECK(cudaDeviceSynchronize());
      start = high_resolution_clock::now();
      uint32_t* SA_S_ptr = thrust::raw_pointer_cast(SA_S.data());
      auto r = thrust::counting_iterator<int>(0);
      thrust::for_each(r, r + s_n,
                       [SA_S_ptr, s_start, reference_sz] __device__(int i) {
                         SA_S_ptr[i] += s_start - reference_sz;
                       });

      CHECK(cudaGetLastError());
      CHECK(cudaDeviceSynchronize());

      thrust::copy(SA_S.begin(), SA_S.end(),
                   SA_S_full.begin() + s_start - reference_sz);

      CHECK(cudaGetLastError());
      CHECK(cudaDeviceSynchronize());
      end = high_resolution_clock::now();
      ms_double = end - start;
      copy_time += ms_double.count();
      chunk_sz = files_at_time;
    }
    std::cout << "\n===***===>\tComputing SAs on GPU of file size: " << n
              << " with S sizes of " << files_at_time << " and chunk factor "
              << chunk_factor << std::endl;
    std::cout << "\n===>\tComputing SAs on GPU took: " << times[0]
              << " milliseconds\n"
              << std::endl;
    std::cout << "\n===>\tComputing LCP on GPU took: " << times[1]
              << " milliseconds\n"
              << std::endl;
    std::cout << "\n===>\tCopyin SA on GPU took: " << copy_time
              << " milliseconds\n"
              << std::endl;
    if (times[0] < best_times[4]) {
      best_times[0] = times[0];
      best_times[1] = times[1];
      best_times[2] = copy_time;
      best_times[4] = times[0];
      best_times[5] = files_at_time;
    }
  }

  std::cout << "\n===>\tBest time on SA took: " << best_times[0]
            << " milliseconds\n"
            << std::endl;
  std::cout << "\n===>\tBest files at time took: " << best_times[5]
            << std::endl;

  std::cout << "\n===>\tBest time on LCP took: "
            << best_times[1] + lcp_time_first << " milliseconds\n"
            << std::endl;

  std::cout << "\n===>\tBest copy time took: " << best_times[2]
            << " milliseconds\n"
            << std::endl;

  times[1] = lcp_time_first;
  start = high_resolution_clock::now();
  thrust::device_vector<uint8_t> S(T.begin() + reference_sz, T.end());
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  end = high_resolution_clock::now();
  ms_double = end - start;
  gpu_copy_time += ms_double.count();  // Time taken to copy S to device
  thrust::device_vector<int> LCP_S(n - reference_sz);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  thrust::device_vector<int> offsets(offs);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  auto lcp_time = compute_LCP_circular_SA(S, SA_S_full, LCP_S, offsets,
                                          n - reference_sz, chunk_sz, 1);
  times[1] += lcp_time;
  int chunk_size = divup(n - reference_sz, chunk_factor);
  int num_chunks = divup(n - reference_sz, chunk_size);
  thrust::device_vector<int> offsets_and_lens(num_chunks * 2);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  start = high_resolution_clock::now();
  get_matching_factor_offsets(SA_R, SA_S_full, R, S, offsets_and_lens,
                              reference_sz, n - reference_sz, chunk_size);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  end = high_resolution_clock::now();
  ms_double = end - start;
  times[2] += ms_double.count();
  std::cout << "\n===> Get offsets and lens took: " << ms_double.count()
            << " milliseconds\n"
            << std::endl;
  // compute the Matching Statistics directly from the SA and the LCP arrays
  thrust::device_vector<int> MS((n - reference_sz) * 2);

  start = high_resolution_clock::now();
  get_factors_directly_circular_R(R, S, SA_R, SA_S_full, LCP_R, LCP_S,
                                  offsets_and_lens, MS, chunk_size);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  end = high_resolution_clock::now();
  ms_double = end - start;
  times[3] += ms_double.count();
  std::cout << "\n===> Compute MS directly from SA and LCP took: "
            << ms_double.count() << " milliseconds\n"
            << std::endl;
  std::cout << "\n===>\tBest time on Offsets took: " << times[2]
            << " milliseconds\n"
            << std::endl;
  std::cout << "\n===>\tBest time on MS took: " << times[3] << " milliseconds\n"
            << std::endl;
  std::vector<int> positions(n - reference_sz);
  std::vector<int> lengths(n - reference_sz);
  std::cout << "\n===>\tCopying took: " << gpu_copy_time << " milliseconds\n"
            << std::endl;
  best_times[4] = best_times[0] + times[1] + times[2] + times[3] +
                  gpu_copy_time + best_times[2];
  std::cout << "\n===>\tBest time on GPU took: " << best_times[4]
            << " milliseconds\n"
            << std::endl;
  std::cout << "\nChunk size: " << chunk_size << ", num_chunks: " << num_chunks
            << ", chunk_factor: " << chunk_factor << std::endl;

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