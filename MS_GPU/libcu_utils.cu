#pragma once

/* #include <libcubwt.cu> */
#include <vector>

#include "utils.cu"
#include "libcubwt.cuh"
#include "libcubwt.cu"

double Compute_SA_libcu(/* uint8_t* */ thrust::device_vector<uint8_t> input_text, thrust::device_vector<uint>& SA,
                        int n, bool log_times = 1) {
  using std::chrono::duration;
  using std::chrono::duration_cast;
  using std::chrono::high_resolution_clock;
  using std::chrono::milliseconds;
  auto start = high_resolution_clock::now();
  void* device_storage;
  int64_t error = libcubwt_allocate_device_storage(&device_storage, n);

  if (error != 0) {
    std::cerr << "Error allocating device storage: " << error << std::endl;
    return EXIT_FAILURE;
  }
  auto end = high_resolution_clock::now();
  duration<double, std::milli> ms_double = end - start;
  if (log_times)
    std::cout << "\nMem alloc and copy to device by "
                 "libcubwt_allocate_device_storage took: "
              << ms_double.count() << " milliseconds\n"
              << std::endl;

  /* thrust::device_vector<uint8_t> T_device(T); */
  /* thrust::device_vector<uint32_t> SA_device_storage(n); */
  uint32_t* sa_device_storage_ptr =
      thrust::raw_pointer_cast(SA/* _device_storage */.data());
  /* uint8_t* T_device_ptr = thrust::raw_pointer_cast(T_device.data()); */
  uint8_t* input_text_ptr = thrust::raw_pointer_cast(input_text.data());
  // start = high_resolution_clock::now();
  error = libcubwt_sa(device_storage, input_text_ptr, sa_device_storage_ptr, n);
  if (error != 0) {
    std::cerr << "Error computing SA: " << error << std::endl;
    return EXIT_FAILURE;
  }
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  end = high_resolution_clock::now();
  ms_double = end - start;
  if (log_times)
    std::cout << "\n===> SA construction by libcubwt_compute_sa took: "
              << ms_double.count() << " milliseconds\n"
              << std::endl;
  /* thrust::copy(SA_device_storage.begin(), SA_device_storage.end(), SA.begin()); */
  //start = high_resolution_clock::now();
  error = libcubwt_free_device_storage(device_storage);
  if (error != 0) {
    std::cerr << "Error freeing device storage: " << error << std::endl;
    return EXIT_FAILURE;
  }
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  end = high_resolution_clock::now();
  ms_double = end - start;
  if (log_times)
    std::cout
        << "\nFreeing device storage by libcubwt_free_device_storage took: "
        << ms_double.count() << " milliseconds\n"
        << std::endl;
  auto sa_time = ms_double.count();
  return sa_time;
}