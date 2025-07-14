#include <chrono>
#include <cstdlib>
#include <iostream>

#include "lcp_utils.cu"
#include "libcu_utils.cu"
#include "merge_utils.cu"
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
  int chunk_factor = SEGSIZE_FACTOR * 16 * 4;
  std::cout << "\nMatching Statistics construction by SA merge, file: "
            << argv[1] << ", size: " << n << " bytes\n"
            << std::endl;

  auto start = high_resolution_clock::now();

  thrust::device_vector<uint8_t> T(input_text);

  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  auto end = high_resolution_clock::now();
  duration<double, std::milli> ms_double = end - start;

  std::cout << "\nMem alloc and copy to device GPU took: " << ms_double.count()
            << " milliseconds\n"
            << std::endl;

  thrust::device_vector<uint8_t> R(T.begin(), T.begin() + reference_sz);
  thrust::device_vector<uint> SA_R(reference_sz);
  thrust::device_vector<int> LCP_R(reference_sz);
  thrust::device_vector<uint8_t> full_MS((n - reference_sz) * 2);

  auto sa_time = Compute_SA_libcu(R, SA_R, reference_sz, 0);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  auto lcp_time = compute_LCP_standard(R, SA_R, LCP_R, reference_sz, 0);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  std::vector<int> positions(n - reference_sz);
  std::vector<int> lengths(n - reference_sz);
  std::vector<double> times(4, 0);  // 0: SA, 2: LCP, 1: offsets, 3: MS
  times[0] = sa_time;
  times[1] = lcp_time;
  auto sa_time_first = sa_time;
  auto lcp_time_first = lcp_time;
  // int files_at_time = 1;
  std::vector<double> best_times(7, 0);
  best_times[4] = (double)INT_MAX;
  for (int files_at_time = 1; files_at_time <= 256; files_at_time *= 2) {
    if (files_at_time > offs.size() - 1) {
      break;
    }
    times[0] = sa_time_first;
    times[1] = lcp_time_first;
    times[2] = 0;
    times[3] = 0;
    double copy_time = 0;
    for (int s_file = 0; s_file < offs.size() - files_at_time;
         s_file += files_at_time) {
      auto s_start = offs[s_file];
      auto s_end = offs[min(s_file + files_at_time, (int)offs.size() - 1)];
      int s_n = s_end - s_start;

      start = high_resolution_clock::now();
      thrust::device_vector<uint8_t> S(T.begin() + s_start, T.begin() + s_end);
      CHECK(cudaGetLastError());
      CHECK(cudaDeviceSynchronize());
      end = high_resolution_clock::now();
      ms_double = end - start;
      copy_time += ms_double.count();

      thrust::device_vector<uint> SA_S(s_n);
      CHECK(cudaGetLastError());
      CHECK(cudaDeviceSynchronize());
      sa_time = Compute_SA_libcu(S, SA_S, s_n, 0);
      times[0] += sa_time;
      CHECK(cudaGetLastError());
      CHECK(cudaDeviceSynchronize());
      thrust::device_vector<int> LCP_S(s_n);
      lcp_time = compute_LCP_standard(S, SA_S, LCP_S, s_n, 0);
      times[1] += lcp_time;
      CHECK(cudaGetLastError());
      CHECK(cudaDeviceSynchronize());

      int chunk_size = divup(s_n, chunk_factor);
      int num_chunks = divup(s_n, chunk_size);
      thrust::device_vector<int> offsets_and_lens(num_chunks * 2);
      CHECK(cudaGetLastError());
      CHECK(cudaDeviceSynchronize());

      start = high_resolution_clock::now();
      get_offsets_and_lens(SA_R, SA_S, R, S, offsets_and_lens, reference_sz,
                           s_n, chunk_size);
      CHECK(cudaGetLastError());
      CHECK(cudaDeviceSynchronize());
      end = high_resolution_clock::now();
      ms_double = end - start;
      times[2] += ms_double.count();
      // compute the Matching Statistics directly from the SA and the LCP arrays
      thrust::device_vector<int> MS(s_n * 2);
      CHECK(cudaGetLastError());
      CHECK(cudaDeviceSynchronize());
      start = high_resolution_clock::now();
      get_factors_directly(R, S, SA_R, SA_S, LCP_R, LCP_S, offsets_and_lens, MS,
                           chunk_size);
      CHECK(cudaGetLastError());
      CHECK(cudaDeviceSynchronize());
      end = high_resolution_clock::now();
      ms_double = end - start;
      times[3] += ms_double.count();
      start = high_resolution_clock::now();
      thrust::copy(full_MS.begin() + s_file * 2,
                   full_MS.begin() + (s_file + files_at_time) * 2, MS.begin());
      CHECK(cudaGetLastError());
      CHECK(cudaDeviceSynchronize());
      end = high_resolution_clock::now();
      ms_double = end - start;
      copy_time += ms_double.count();
    }
    std::cout << "\n===***===>\tComputing SAs on GPU of file size: " << n
              << " with S sizes of " << files_at_time << " and chunk factor "
              << chunk_factor << std::endl;
    std::cout << "\n===>\tComputing SAs took: " << times[0] << " milliseconds\n"
              << std::endl;
    std::cout << "\n===>\tComputing LCPs took: " << times[1]
              << " milliseconds\n"
              << std::endl;
    std::cout << "\n===>\tComputing Offsets took: " << times[2]
              << " milliseconds\n"
              << std::endl;
    std::cout << "\n===>\tComputing MS took: " << times[3] << " milliseconds\n"
              << std::endl;
    std::cout << "\n===>\tTotal time took: "
              << times[0] + times[1] + times[2] + times[3] << " milliseconds\n"
              << std::endl;
    std::cout << "\n===>\tGPU copy time: " << copy_time << std::endl;
    std::cout << offs.size() << std::endl;
    if (times[0] + times[1] + times[2] + times[3] + copy_time < best_times[4]) {
      best_times[0] = times[0];
      best_times[1] = times[1];
      best_times[2] = times[2];
      best_times[3] = times[3];
      best_times[4] = times[0] + times[1] + times[2] + times[3] + copy_time;
      best_times[5] = files_at_time;
      best_times[6] = copy_time;
    }
  }
  std::cout << "\n===>\tBest time on GPU took: " << best_times[4]
            << " milliseconds\n"
            << std::endl;

  std::cout << "\n===>\tBest time on SA took: " << best_times[0]
            << " milliseconds\n"
            << std::endl;
  std::cout << "\n===>\tBest time on LCP took: " << best_times[1]
            << " milliseconds\n"
            << std::endl;
  std::cout << "\n===>\tBest time on Offsets took: " << best_times[2]
            << " milliseconds\n"
            << std::endl;
  std::cout << "\n===>\tBest time on MS took: " << best_times[3]
            << " milliseconds\n"
            << std::endl;
  std::cout << "\n===>\tBest files at time took: " << best_times[5]
            << std::endl;
  std::cout << "\n===>\tBest GPU copy time took: " << best_times[6]
            << " milliseconds\n"
            << std::endl;
}