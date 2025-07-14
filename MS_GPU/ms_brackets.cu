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
  /* n = reference_sz * 2 + 1;
  input_text.resize(n);
  input_text[n - 1] = '$'; */

  std::cout << "\nMatching Statistics construction, file: " << argv[1]
            << ", size: " << n << " bytes\n"
            << std::endl;

  auto start = high_resolution_clock::now();

  thrust::device_vector<uint8_t> T(input_text);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  thrust::device_vector<uint> SA(n);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  auto end = high_resolution_clock::now();
  duration<double, std::milli> ms_double = end - start;
  std::cout << "\nMem alloc and copy to device GPU took: " << ms_double.count()
            << " milliseconds\n"
            << std::endl;

  auto sa_time = Compute_SA_libcu(input_text, SA, n);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  std::cout << "\nSA construction on GPU of file size: " << n
            << " took: " << sa_time << " milliseconds\n"
            << std::endl;
  std::cout << "SA construction time: " << sa_time << " seconds\n" << std::endl;
  auto total_time = sa_time;
  thrust::device_vector<int> l_flags(n);  // array to mark left types
  thrust::device_vector<int> l_ranks(n);  // ranks of the left types
  thrust::device_vector<int> r_flags(n);  // array to mark right types
  thrust::device_vector<int> r_ranks(n);  // ranks of the right types
  thrust::device_vector<int> M_vector(n * 2);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  start = high_resolution_clock::now();

  MarkLeftAndRightTypes(SA, l_flags, r_flags, n, reference_sz - 1);
  UpdateRanks(l_flags, l_ranks, n);
  UpdateRanks(r_flags, r_ranks, n);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  end = high_resolution_clock::now();

  ms_double = end - start;
  total_time += ms_double.count();
  std::cout
      << "\nBracket marking and stream compaction of brackets on GPU took: "
      << ms_double.count() << " milliseconds\n"
      << std::endl;

  start = high_resolution_clock::now();
  thrust::device_vector<int>::iterator it_l = l_ranks.end() - 1;
  thrust::device_vector<int>::iterator it_r = r_ranks.end() - 1;
  int itl = *it_l;
  int itr = *it_r;
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  end = high_resolution_clock::now();
  ms_double = end - start;
  std::cout << "\n#l_ranks: " << itl << ", #r_ranks: " << itr
            << "\nretrieving these numbers took " << ms_double.count()
            << " milliseconds\n"
            << std::endl;
  std::printf("reference_size  %d \n", reference_sz);

  total_time += ms_double.count();
  start = high_resolution_clock::now();
  thrust::device_vector<int> l_vec(
      itl, -1);  // to contain the SA index for the ith l_type
  thrust::device_vector<int> r_vec(
      itl, -1);  // to contain the SA index for the ith r_type

  PopulateVec(l_vec, l_flags, l_ranks, SA, n);
  PopulateVec(r_vec, r_flags, r_ranks, SA, n);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  end = high_resolution_clock::now();
  ms_double = end - start;

  total_time += ms_double.count();
  std::cout << "\nPopulating 2 vectors took: " << ms_double.count()
            << " milliseconds\n"
            << std::endl;

    start = high_resolution_clock::now();
    // let's convert the flag arrays to become M_l and M_r for now
    ComputeMs(SA, l_vec, r_vec, l_flags, r_flags, l_ranks, r_ranks, n);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    end = high_resolution_clock::now();
    ms_double = end - start;
    std::cout << "\nComputeMs took: " << ms_double.count() << " milliseconds\n"
              << std::endl;
  start = high_resolution_clock::now();
  ComputeMs_to_one_array(SA, l_vec, r_vec, M_vector, l_ranks, r_ranks, n);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  end = high_resolution_clock::now();
  ms_double = end - start;
  std::cout << "\nComputeMs_to_one_array took: " << ms_double.count()
            << " milliseconds\n"
            << std::endl;

  total_time += ms_double.count();

   SA.clear();
   SA.shrink_to_fit();
   CHECK(cudaGetLastError());
   CHECK(cudaDeviceSynchronize());

   start = high_resolution_clock::now();
   // write the l_lcp to l_rank array
   compute_LCP(T, l_flags, l_ranks, n, reference_sz - 1);
   // write the r_lcp to r_rank array
   compute_LCP(T, r_flags, r_ranks, n, reference_sz - 1);

   // The MS values are written as follows: l_ranks contain the factor positions
   // and r_ranks contain the lengths of matching factors
   get_MS_values(l_flags, r_flags, l_ranks, r_ranks, n, reference_sz);
   end = high_resolution_clock::now();
   ms_double = end - start;
   std::cout
       << "\nMS construction by two lcp array contruction plus one comparisment"
          " took: "
       << ms_double.count() << " milliseconds\n"
       << std::endl;

   start = high_resolution_clock::now();
   // l_flags contains the positions and r_flags contains the lengths
   compute_LCP_from_both(T, l_flags, r_flags, n, reference_sz);
   CHECK(cudaGetLastError());
   CHECK(cudaDeviceSynchronize());
   end = high_resolution_clock::now();
   ms_double = end - start;
   std::cout << "\nMS construction directly from the M arrays took: "
             << ms_double.count() << " milliseconds\n"
             << std::endl;

   std::cout << "\nComparing lengths\n" << std::endl;
   compare_two_arrays(r_flags, r_ranks, n, reference_sz);
   std::cout << "\nComparing positions\n" << std::endl;
   compare_two_arrays(l_flags, l_ranks, n, reference_sz);
   //  copy data to host

  start = high_resolution_clock::now();

  // writes MS values to the M_vector, [id * 2] contains the position and
  // [id * 2 + 1] contains the length of the matching factor for the
  // sufffix R+S[id]
  compute_LCP_from_one_array_directly(T, M_vector, n, reference_sz);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  end = high_resolution_clock::now();
  ms_double = end - start;
  std::cout << "\nMS construction directly from one M array took: "
            << ms_double.count() << " milliseconds\n"
            << std::endl;

  total_time += ms_double.count();

  std::cout << "\nTotal time for MS construction: " << total_time
            << " milliseconds\n"
            << std::endl;
  /*  std::cout << "\nComparing lengths from one M array\n" << std::endl;
   compare_two_arrays(M_vector, r_ranks, n, reference_sz, 1);

   std::cout << "\nComparing positions from one M array\n" << std::endl;
   compare_two_arrays(M_vector, l_ranks, n, reference_sz, 1, 1);
  */
  std::vector<int> positions(n - reference_sz);
  std::vector<int> lengths(n - reference_sz);
  start = high_resolution_clock::now();

  thrust::copy(l_ranks.begin() + reference_sz, l_ranks.end(),
               positions.begin());

  thrust::copy(r_ranks.begin() + reference_sz, r_ranks.end(), lengths.begin());
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  end = high_resolution_clock::now();
  ms_double = end - start;
  std::cout << "\nData transfer of full MS "
            << " of size: " << lengths.size() * 2 << " bytes, took "
            << ms_double.count() << " milliseconds\n"
            << std::endl;
  start = high_resolution_clock::now();
  mark_CMS_heads(r_ranks, l_flags, n, reference_sz);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  // use flags arrays for ranks, since len and pos are written to the ranks
  // arrays
  UpdateRanks(l_flags, r_flags, n);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  end = high_resolution_clock::now();
  ms_double = end - start;
  std::cout << "\nMarking CMS heads + prefix sums took " << ms_double.count()
            << " milliseconds\n"
            << std::endl;
  auto heads_get = ms_double.count();
  start = high_resolution_clock::now();
  thrust::device_vector<int>::iterator it = r_flags.end() - 1;
  int num_heads = *it;
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  end = high_resolution_clock::now();
  ms_double = end - start;
  heads_get += ms_double.count();
  std::cout << "\n#heads: " << num_heads << " out of " << n - reference_sz
            << ", (" << (float)num_heads / (n - reference_sz)
            << ") values, retrieving this number took " << ms_double.count()
            << " milliseconds\n"
            << std::endl;


  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  start = high_resolution_clock::now();

  write_CMS_values(M_vector, l_flags, r_flags, l_ranks, r_ranks, n);
  end = high_resolution_clock::now();
  ms_double = end - start;
  std::cout << "\nwriting CMS values took " << ms_double.count()
            << " milliseconds\n"
            << std::endl;
  heads_get += ms_double.count();
  std::vector<int> cms((size_t)3 * num_heads);
  start = high_resolution_clock::now();

  thrust::copy(M_vector.begin(), M_vector.begin() + cms.size(), cms.begin());

  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  end = high_resolution_clock::now();
  ms_double = end - start;
  heads_get += ms_double.count();
  std::cout << "\nComputing CMS of size: " << cms.size() * sizeof(int)
            << " bytes, took " << heads_get - ms_double.count()
            << " milliseconds\n"
            << std::endl;
  std::cout << "\nTotal time for CMS heads marking and data transfer: "
            << heads_get << " milliseconds\n"
            << std::endl;
  std::cout << "\nData transfer of CMS took " << ms_double.count()
            << " milliseconds\n"
            << std::endl;
  // write .ms to file in binary
  auto out_file = std::string(argv[1]) + ".GPU.ms";

  /* std::ofstream ofs(out_file, std::ios::binary);

  for (size_t i = 0; i < n - reference_sz; ++i)
  {
    ofs.write((char *)&positions[i], sizeof(decltype(positions)::value_type));
    ofs.write((char *)&lengths[i], sizeof(decltype(lengths)::value_type));
    // std::cout << positions[i] << "," << lengths[i] << std::endl;
  }
 */
  /* write the SA to file
  ofs.write(reinterpret_cast<char*>(SA_host.data()),
            sizeof(decltype(SA_host)::value_type) * SA_host.size());
  ofs.close(); */
}