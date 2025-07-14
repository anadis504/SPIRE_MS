#include <chrono>
#include <cstdlib>
#include <iostream>

#include "libcu_utils.cu"
#include "ms_utils.cu"
#include "utils.cu"

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::fprintf(
        stderr,
        "usage: %s [input text file] [(optional) write SA file: 1, !1 (debug)"
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

  auto start = high_resolution_clock::now();

  thrust::device_vector<uint8_t> T(input_text);

  auto end = high_resolution_clock::now();
  duration<double, std::milli> ms_double = end - start;

  std::cout << "\nMem alloc and copy of T to device (thrust) took: "
            << ms_double.count() << " milliseconds\n"
            << std::endl;
  std::vector<double> best_times(7, 0);  // 0: SA, 2: 5: files_at time
  best_times[4] = (double)INT_MAX;
  best_times[0] = (double)INT_MAX;

  for (int concat_num = 2; concat_num < 16 * 16 * 16; concat_num *= 2) {
    if (concat_num > offs.size() * 2 - 1) break;
    std::vector<double> times(4, 0);  // 0: SA, 1: preprocessing, 2:
    // compute_M_vec, 3: MS

    int rounds = divup(offs.size() - 1, concat_num - 1);
    double gpu_copy_time = 0;
    for (int round = 0; round < rounds; ++round) {
      int first_doc_ind = round * (concat_num - 1);
      int last_doc_ind = first_doc_ind + (concat_num - 1);
      last_doc_ind = min(last_doc_ind, (int)offs.size() - 1);

      if (first_doc_ind >= last_doc_ind) break;
      auto s_start = offs[first_doc_ind];
      auto s_end = offs[last_doc_ind];

      size_t current_n =
          offs[last_doc_ind] - offs[first_doc_ind] + reference_sz;

      start = high_resolution_clock::now();

      thrust::device_vector<uint8_t> current_T(T.begin(),
                                               T.begin() + reference_sz);
      current_T.insert(current_T.begin() + reference_sz, T.begin() + s_start,
                       T.begin() + s_end);
      CHECK(cudaGetLastError());
      CHECK(cudaDeviceSynchronize());
      end = high_resolution_clock::now();
      ms_double = end - start;
      gpu_copy_time += ms_double.count();

      CHECK(cudaGetLastError());
      CHECK(cudaDeviceSynchronize());
      thrust::device_vector<uint> SA_RS(current_n);
      CHECK(cudaGetLastError());
      CHECK(cudaDeviceSynchronize());
      auto sa_time = Compute_SA_libcu(current_T, SA_RS, current_n, 0);
      times[0] += sa_time;
      CHECK(cudaGetLastError());
      CHECK(cudaDeviceSynchronize());

      thrust::device_vector<int> l_flags(
          current_n);  // array to mark left types
      thrust::device_vector<int> l_ranks(current_n);  // ranks of the left types
      thrust::device_vector<int> r_flags(
          current_n);  // array to mark right types
      thrust::device_vector<int> r_ranks(
          current_n);  // ranks of the right types
      thrust::device_vector<int> M_vector(current_n * 2);
      CHECK(cudaGetLastError());
      CHECK(cudaDeviceSynchronize());
      start = high_resolution_clock::now();

      MarkLeftAndRightTypes(SA_RS, l_flags, r_flags, current_n,
                            reference_sz - 1);
      UpdateRanks(l_flags, l_ranks, current_n);
      UpdateRanks(r_flags, r_ranks, current_n);
      CHECK(cudaGetLastError());
      CHECK(cudaDeviceSynchronize());
      thrust::device_vector<int>::iterator it_l = l_ranks.end() - 1;
      thrust::device_vector<int>::iterator it_r = r_ranks.end() - 1;
      int itl = *it_l;
      int itr = *it_r;
      CHECK(cudaGetLastError());
      CHECK(cudaDeviceSynchronize());
      thrust::device_vector<int> l_vec(
          itl, -1);  // to contain the SA index for the ith l_type
      thrust::device_vector<int> r_vec(
          itl, -1);  // to contain the SA index for the ith r_type

      PopulateVec(l_vec, l_flags, l_ranks, SA_RS, current_n);
      PopulateVec(r_vec, r_flags, r_ranks, SA_RS, current_n);
      CHECK(cudaGetLastError());
      CHECK(cudaDeviceSynchronize());
      end = high_resolution_clock::now();

      ms_double = end - start;
      times[1] += ms_double.count();

      start = high_resolution_clock::now();
      ComputeMs_to_one_array(SA_RS, l_vec, r_vec, M_vector, l_ranks, r_ranks,
                             current_n);
      CHECK(cudaGetLastError());
      CHECK(cudaDeviceSynchronize());
      end = high_resolution_clock::now();
      ms_double = end - start;
      times[2] += ms_double.count();
      /* std::cout << "\n===>\tComputing M_vector took: " << ms_double.count()
      << " milliseconds\n"
      << std::endl; */

      start = high_resolution_clock::now();
      compute_LCP_from_one_array_directly(current_T, M_vector, current_n,
                                          reference_sz);
      CHECK(cudaGetLastError());
      CHECK(cudaDeviceSynchronize());
      end = high_resolution_clock::now();
      ms_double = end - start;
      times[3] += ms_double.count();
    }
    std::cout << "\n===***===>\tComputing SAs on GPU of file size: " << n
              << " with S concats sizes of " << concat_num << std::endl;
    std::cout << "\n===>\tComputing SAs on GPU of file size: " << n
              << " took: " << times[0] << " milliseconds\n"
              << std::endl;
    std::cout << "\n===>\tMarking left and right types, updating ranks and "
                 "populating vectors took: "
              << times[1] << " milliseconds\n"
              << std::endl;
    std::cout << "\n===>\tComputing M_vector took: " << times[2]
              << " milliseconds\n"
              << std::endl;
    std::cout << "\n===>\tComputing MS took: " << times[3] << " milliseconds\n"
              << std::endl;

    std::cout << "\n===>\tCopyin time gpu alltogether: " << n
              << " * 4 bytes took: " << gpu_copy_time << " milliseconds\n"
              << std::endl;

    if (times[0] < best_times[0]) {
      best_times[0] = times[0];
      best_times[1] = times[1];
      best_times[2] = times[2];
      best_times[3] = times[3];
      best_times[4] = times[0] + times[1] + times[2] + times[3] + gpu_copy_time;
      best_times[5] = concat_num;
      best_times[6] = gpu_copy_time;
    }
  }
  std::cout << "\n===>\tBest time on SA took: " << best_times[0]
            << " milliseconds\n"
            << std::endl;
  std::cout << "\n===>\tBest files at time took: " << best_times[5]
            << std::endl;
  std::cout << "\n===>\tBest time on preprocessing took: " << best_times[1]
            << " milliseconds\n"
            << std::endl;
  std::cout << "\n===>\tBest time on M_vec took: " << best_times[2]
            << " milliseconds\n"
            << std::endl;
  std::cout << "\n===>\tBest time on MS took: " << best_times[3]
            << " milliseconds\n"
            << std::endl;
  std::cout << "\n===>\tBest time on copying to GPU took: " << best_times[6]
            << " milliseconds\n"
            << std::endl;
  std::cout << "\n===>\tBest time alltogether  took: " << best_times[4]
            << " milliseconds\n"
            << std::endl;
}