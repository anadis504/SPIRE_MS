#pragma once
#include "utils.cu"

void compute_phi(thrust::device_vector<uint32_t>& sa,
                 thrust::device_vector<int32_t>& phi, size_t n) {
  uint32_t* sa_ptr = thrust::raw_pointer_cast(sa.data());
  int32_t* phi_ptr = thrust::raw_pointer_cast(phi.data());
  auto r = thrust::counting_iterator<int32_t>(0);
  thrust::for_each(r, r + n, [=] __device__(int32_t i) {
    if (i == 0) {
      phi_ptr[sa_ptr[i]] = -1;
    } else {
      phi_ptr[sa_ptr[i]] = sa_ptr[i - 1];
    }
  });
}

void mark_offsets_in_circular_SA(thrust::device_vector<uint32_t>& sa,
                                 thrust::device_vector<int32_t>& phi,
                                 thrust::device_vector<int32_t>& offsets,
                                 size_t offs, size_t chunk_sz) {
  uint32_t* sa_ptr = thrust::raw_pointer_cast(sa.data());
  int32_t* phi_ptr = thrust::raw_pointer_cast(phi.data());
  int32_t* offsets_ptr = thrust::raw_pointer_cast(offsets.data());
  auto r = thrust::counting_iterator<int32_t>(0);
  thrust::for_each(r, r + offs/chunk_sz, [=] __device__(int32_t i) {
    if (i * chunk_sz >= offs-1) return;
    int32_t pos = offsets_ptr[i*chunk_sz] - offsets_ptr[0];
    phi_ptr[sa_ptr[pos]] = -1;
  });
}

__global__ void compute_PLCP_kernel(uint8_t* t, int32_t* phi_plcp, size_t n,
                                    size_t seg_size) {
  int32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  tid *= seg_size;
  if (tid >= n) return;
  int32_t h = 0;
  for (int32_t i = 0; i < seg_size; ++i) {
    if (tid + i < n) {
      if (phi_plcp[tid + i] == -1) {
        h = 0;
      } else {
        int32_t k = phi_plcp[tid + i];
        while (tid + i + h < n && k + h < n && t[tid + i + h] == t[k + h]) {
          ++h;
        }
      }
      phi_plcp[tid + i] = h;
      /* if (!h) {
        printf("h: %d, tid + i: %d, t[tid + i]: %c\n", h, tid + i,
               (char)t[tid + i]);
      } */
      if (h) {
        --h;
      }
    }
  }
}

void compute_PLCP(thrust::device_vector<uint8_t>& T,
                  thrust::device_vector<int32_t>& phi_plcp, size_t n,
                  size_t block_size, size_t bl_num, size_t seg_size) {
  uint8_t* t_ptr = thrust::raw_pointer_cast(T.data());
  int32_t* phi_plcp_ptr = thrust::raw_pointer_cast(phi_plcp.data());

  compute_PLCP_kernel<<<bl_num, block_size>>>(t_ptr, phi_plcp_ptr, n, seg_size);

  CHECK(cudaGetLastError());
  cudaDeviceSynchronize();
}

void unpermute_PLCP(thrust::device_vector<uint32_t>& sa,
                    thrust::device_vector<int32_t>& plcp,
                    thrust::device_vector<int32_t>& lcp, size_t n) {
  uint32_t* sa_ptr = thrust::raw_pointer_cast(sa.data());
  int32_t* plcp_ptr = thrust::raw_pointer_cast(plcp.data());
  int32_t* lcp_ptr = thrust::raw_pointer_cast(lcp.data());
  auto r = thrust::counting_iterator<int32_t>(0);
  thrust::for_each(r, r + n, [=] __device__(int32_t i) {
    lcp_ptr[i] = plcp_ptr[sa_ptr[i]];
  });
}

void compute_ISA(thrust::device_vector<uint>& sa,
                 thrust::device_vector<uint>& isa, size_t n) {
  uint* sa_ptr = thrust::raw_pointer_cast(sa.data());
  uint* isa_ptr = thrust::raw_pointer_cast(isa.data());
  auto r = thrust::counting_iterator<int>(0);
  thrust::for_each(r, r + n, [=] __device__(int i) { isa_ptr[sa_ptr[i]] = i; });
}

__global__ void compute_kasai_LCP_kernel(uint8_t* t, uint32_t* sa,
                                         uint32_t* isa, int32_t* lcp, size_t n,
                                         size_t seg_size) {
  int32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  tid *= seg_size;
  if (tid >= n) return;
  int32_t h = 0;
  for (int32_t i = 0; i < seg_size; ++i) {
    if (tid + i < n) {
      if (isa[tid + i]) {
        int32_t k = sa[isa[tid + i] - 1];
        while (tid + i + h < n && k + h < n && t[tid + i + h] == t[k + h]) {
          ++h;
        }
      } else {
        h = 0;
      }
      lcp[isa[tid + i]] = h;
      if (h) {
        --h;
      }
    }
  }
}

void compute_kasai_LCP(thrust::device_vector<uint8_t>& T,
                       thrust::device_vector<uint32_t>& sa,
                       thrust::device_vector<uint32_t>& isa,
                       thrust::device_vector<int32_t>& lcp, size_t n,
                       size_t block_size, size_t bl_num, size_t seg_size) {
  uint8_t* t_ptr = thrust::raw_pointer_cast(T.data());
  uint32_t* sa_ptr = thrust::raw_pointer_cast(sa.data());
  uint32_t* isa_ptr = thrust::raw_pointer_cast(isa.data());
  int32_t* lcp_ptr = thrust::raw_pointer_cast(lcp.data());

  compute_kasai_LCP_kernel<<<bl_num, block_size>>>(t_ptr, sa_ptr, isa_ptr,
                                                   lcp_ptr, n, seg_size);

  CHECK(cudaGetLastError());
  cudaDeviceSynchronize();
}

bool check_LCP(thrust::device_vector<uint8_t>& T,
               thrust::device_vector<int32_t>& lcp,
               thrust::device_vector<uint32_t>& sa, size_t n) {
  uint8_t* t_ptr = thrust::raw_pointer_cast(T.data());
  int32_t* lcp_ptr = thrust::raw_pointer_cast(lcp.data());
  uint32_t* sa_ptr = thrust::raw_pointer_cast(sa.data());
  thrust::device_vector<bool> wrongs(n);
  bool* W_ptr = thrust::raw_pointer_cast(wrongs.data());
  auto r = thrust::counting_iterator<int32_t>(1);
  thrust::for_each(r, r + n, [=] __device__(int32_t i) {
    int32_t k = sa_ptr[i - 1];
    int32_t j = sa_ptr[i];
    int lcp = lcp_ptr[i];
    W_ptr[i] = 0;

    for (int32_t h = 0; h < lcp; ++h) {
      if (t_ptr[k + h] != t_ptr[j + h]) {
        /* printf("LCP check failed at index %d: %d != %d\n", i, t_ptr[k + h],
               t_ptr[j + h]); */
        W_ptr[i] = 1;
        return;
      }
    }
    if (k + lcp < n && j + lcp < n && t_ptr[k + lcp] == t_ptr[j + lcp]) {
      W_ptr[i] = 1;
      /* printf("LCP check failed at index %d: %d == %d, lcp = %d\n", i,
             t_ptr[k + lcp], t_ptr[j + lcp], lcp); */
      return;
    }
  });
  int32_t wrongs_count = thrust::reduce(wrongs.begin(), wrongs.end(), 0);
  if (wrongs_count > 0) {
    printf("LCP check failed: %d wrongs of %d values\n", wrongs_count, n);
    return false;
  }
  return true;
}

double compute_lcp_avg(thrust::device_vector<int32_t>& lcp, size_t n) {
  using std::chrono::duration;
  using std::chrono::duration_cast;
  using std::chrono::high_resolution_clock;
  using std::chrono::milliseconds;
  auto start = high_resolution_clock::now();
  int s_sz = 100000;
  auto s = divup(n, s_sz);

  double avg = 0.0;
  for (int i = 0; i < s; ++i) {
    size_t sum = thrust::reduce(lcp.begin() + i * s_sz,
                                lcp.begin() + min(size_t(i + 1) * s_sz, n), 0);
    CHECK(cudaGetLastError());
    cudaDeviceSynchronize();
    auto local =
        static_cast<double>(sum) / min(size_t(n - i * s_sz), (size_t)s_sz);
    avg = (avg + local) / 2;
    /*  std::cout << "Sum of LCP segment " << i * s_sz << ": " << sum
               << " average : " << avg << std::endl; */
  }

  return avg;
}

double compute_LCP_standard(thrust::device_vector<uint8_t>& T,
                            thrust::device_vector<uint32_t>& sa,
                            thrust::device_vector<int32_t>& lcp, size_t n,
                            bool log_times = true) {
  using std::chrono::duration;
  using std::chrono::duration_cast;
  using std::chrono::high_resolution_clock;
  using std::chrono::milliseconds;
  auto start = high_resolution_clock::now();
  thrust::device_vector<int32_t> phi_plcp(n);
  compute_phi(sa, phi_plcp, n);
  CHECK(cudaGetLastError());
  cudaDeviceSynchronize();
  int32_t block_size = WARP_SIZE;
  int seg_size = divup(n, SEGSIZE_FACTOR);
  int32_t bl_num = divup(n, block_size * seg_size);
  compute_PLCP(T, phi_plcp, n, block_size, bl_num, seg_size);
  CHECK(cudaGetLastError());
  cudaDeviceSynchronize();
  unpermute_PLCP(sa, phi_plcp, lcp, n);
  CHECK(cudaGetLastError());
  cudaDeviceSynchronize();
  auto end = high_resolution_clock::now();
  duration<double, std::milli> ms_double = end - start;
  if (log_times)
    std::cout << "\n===>\tComputing LCP on GPU of file size: " << n
              << " took: " << ms_double.count() << " milliseconds\n"
              << std::endl;

  auto avg = compute_lcp_avg(lcp, n);
  std::cout << "\n===>\tAverage LCP on GPU of file size: " << n
            << " is: " << avg << "\n"
            << std::endl;
  auto max_lcp =
      thrust::reduce(lcp.begin(), lcp.end(), 0, thrust::maximum<int32_t>());
  std::cout << "\n===>\tMax LCP on GPU of file size: " << n
            << " is: " << max_lcp << "\n"
            << std::endl;
  return ms_double.count();
  /*  start = high_resolution_clock::now();
  bool pass = check_LCP(T, lcp, sa, n);
  CHECK(cudaGetLastError());
  cudaDeviceSynchronize();
  end = high_resolution_clock::now();
  ms_double = end - start;
  /* std::cout << "\n\tLCP check on GPU of file size: " << n
  << " took: " << ms_double.count() << " milliseconds\n"
  << std::endl; */
  /* if (pass) {
  std::cout << "--- LCP check passed" << std::endl;
  } else {
  std::cerr << "--- LCP check failed" << std::endl;
  } */
}

double compute_LCP_circular_SA(thrust::device_vector<uint8_t>& T,
                               thrust::device_vector<uint32_t>& sa,
                               thrust::device_vector<int32_t>& lcp,
                               thrust::device_vector<int32_t>& offsets,
                               size_t n, size_t chunk_sz, bool log_times = true) {
  using std::chrono::duration;
  using std::chrono::duration_cast;
  using std::chrono::high_resolution_clock;
  using std::chrono::milliseconds;
  auto start = high_resolution_clock::now();
  thrust::device_vector<int32_t> phi_plcp(n);
  compute_phi(sa, phi_plcp, n);
  CHECK(cudaGetLastError());
  cudaDeviceSynchronize();
  mark_offsets_in_circular_SA(sa, phi_plcp, offsets, offsets.size(), chunk_sz);
  CHECK(cudaGetLastError());
  cudaDeviceSynchronize();
  int32_t block_size = WARP_SIZE;
  int seg_size = divup(n, SEGSIZE_FACTOR);
  int32_t bl_num = divup(n, block_size * seg_size);
  compute_PLCP(T, phi_plcp, n, block_size, bl_num, seg_size);
  CHECK(cudaGetLastError());
  cudaDeviceSynchronize();
  unpermute_PLCP(sa, phi_plcp, lcp, n);
  CHECK(cudaGetLastError());
  cudaDeviceSynchronize();
  auto end = high_resolution_clock::now();
  duration<double, std::milli> ms_double = end - start;
  if (log_times)
    std::cout << "\n===>\tComputing LCP on GPU of file size: " << n
              << " took: " << ms_double.count() << " milliseconds\n"
              << std::endl;

  auto avg = compute_lcp_avg(lcp, n);
  std::cout << "\n===>\tAverage LCP on GPU of file size: " << n
            << " is: " << avg << "\n"
            << std::endl;
  auto max_lcp =
      thrust::reduce(lcp.begin(), lcp.end(), 0, thrust::maximum<int32_t>());
  std::cout << "\n===>\tMax LCP on GPU of file size: " << n
            << " is: " << max_lcp << "\n"
            << std::endl;
  return ms_double.count();
  /* start = high_resolution_clock::now();
  bool pass = check_LCP(T, lcp, sa, n);
  CHECK(cudaGetLastError());
  cudaDeviceSynchronize();
  end = high_resolution_clock::now();
  ms_double = end - start;
  std::cout << "\n\tLCP check on GPU of file size: " << n
            << " took: " << ms_double.count() << " milliseconds\n"
            << std::endl;
  if (pass) {
    std::cout << "--- LCP check passed" << std::endl;
  } else {
    std::cerr << "--- LCP check failed" << std::endl;
  } */
}
