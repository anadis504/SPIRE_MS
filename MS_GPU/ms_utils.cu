#pragma once

#include "utils.cu"
// #define SEGSIZE 10000

void MarkLeftAndRightTypes(thrust::device_vector<uint>& SA,
                           thrust::device_vector<int>& l_flags,
                           thrust::device_vector<int>& r_flags, int n,
                           size_t dict_end) {
  uint* SA_ptr = thrust::raw_pointer_cast(SA.data());
  int* l_flags_ptr = thrust::raw_pointer_cast(l_flags.data());
  int* r_flags_ptr = thrust::raw_pointer_cast(r_flags.data());
  auto r = thrust::counting_iterator<int>(0);
  // NOTE: do we start checking for r/l types from SA[0], thus going through all
  // the $ entries first or skip the first |docs| entries?
  thrust::for_each(r, r + n, [=] __device__(int i) {
    // mark them all as zeroes
    l_flags_ptr[i] = 0;
    r_flags_ptr[i] = 0;
    // printf("Checking SA[%d]: %d\n", i, SA_ptr[i]);
    // then check if they are left or right types
    if (SA_ptr[i] <= dict_end) {  // this is a dictionary entry
      if (i < n - 1 && SA_ptr[i + 1] > dict_end) {  // is a left type dict entry
        l_flags_ptr[i] = 1;
        // printf("Left type: %d\n", SA_ptr[i]);
      }
      if (i > 0 && SA_ptr[i - 1] > dict_end) {  // is a right type dict entry
        r_flags_ptr[i] = 1;
        // printf("Right type: %d\n", SA_ptr[i]);
      }
    }
  });
}

void UpdateRanks(thrust::device_vector<int>& flags,
                 thrust::device_vector<int>& ranks, int n) {
  int* flags_ptr = thrust::raw_pointer_cast(flags.data());
  int* ranks_ptr = thrust::raw_pointer_cast(ranks.data());
  void* d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, flags_ptr,
                                ranks_ptr, n);

  cudaMalloc(&d_temp_storage, temp_storage_bytes);

  cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, flags_ptr,
                                ranks_ptr, n);

  cudaFree(d_temp_storage);
}

void PopulateVec(thrust::device_vector<int>& vec,
                 thrust::device_vector<int>& flags,
                 thrust::device_vector<int>& ranks,
                 thrust::device_vector<uint>& SA, int n) {
  int* vec_ptr = thrust::raw_pointer_cast(vec.data());
  int* flags_ptr = thrust::raw_pointer_cast(flags.data());
  int* ranks_ptr = thrust::raw_pointer_cast(ranks.data());
  uint* SA_ptr = thrust::raw_pointer_cast(SA.data());
  auto r = thrust::counting_iterator<int>(0);
  thrust::for_each(r, r + n, [=] __device__(int i) {
    if (flags_ptr[i] == 1) {
      vec_ptr[ranks_ptr[i]] = SA_ptr[i];
    }
  });
}

void ComputeMs(thrust::device_vector<uint>& SA,
               thrust::device_vector<int>& l_vec,
               thrust::device_vector<int>& r_vec,
               thrust::device_vector<int>& l_flags,
               thrust::device_vector<int>& r_flags,
               thrust::device_vector<int>& l_ranks,
               thrust::device_vector<int>& r_ranks, int n) {
  uint* SA_ptr = thrust::raw_pointer_cast(SA.data());
  int* l_vec_ptr = thrust::raw_pointer_cast(l_vec.data());
  int* r_vec_ptr = thrust::raw_pointer_cast(r_vec.data());
  int* l_flags_ptr = thrust::raw_pointer_cast(l_flags.data());
  int* r_flags_ptr = thrust::raw_pointer_cast(r_flags.data());
  int* l_ranks_ptr = thrust::raw_pointer_cast(l_ranks.data());
  int* r_ranks_ptr = thrust::raw_pointer_cast(r_ranks.data());
  auto r = thrust::counting_iterator<int>(0);
  thrust::for_each(r, r + n, [=] __device__(int i) {
    l_flags_ptr[SA_ptr[i]] = l_vec_ptr[l_ranks_ptr[i]];
    r_flags_ptr[SA_ptr[i]] = r_vec_ptr[r_ranks_ptr[i] + 1];
  });
}

void ComputeMs_to_one_array(thrust::device_vector<uint>& SA,
                            thrust::device_vector<int>& l_vec,
                            thrust::device_vector<int>& r_vec,
                            thrust::device_vector<int>& M,
                            thrust::device_vector<int>& l_ranks,
                            thrust::device_vector<int>& r_ranks, int n) {
  uint* SA_ptr = thrust::raw_pointer_cast(SA.data());
  int* l_vec_ptr = thrust::raw_pointer_cast(l_vec.data());
  int* r_vec_ptr = thrust::raw_pointer_cast(r_vec.data());
  int* M_ptr = thrust::raw_pointer_cast(M.data());
  int* l_ranks_ptr = thrust::raw_pointer_cast(l_ranks.data());
  int* r_ranks_ptr = thrust::raw_pointer_cast(r_ranks.data());
  auto r = thrust::counting_iterator<int>(0);
  thrust::for_each(r, r + n, [=] __device__(int i) {
    M_ptr[(size_t)SA_ptr[i] * 2] = l_vec_ptr[l_ranks_ptr[i]];
    M_ptr[(size_t)SA_ptr[i] * 2 + 1] = r_vec_ptr[r_ranks_ptr[i] + 1];
  });
}

__global__ void compute_PLCP_kernel(uint8_t* t, int* M, int* lcp, size_t n,
                                    size_t seg_size, size_t offset) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  tid *= seg_size;
  tid += offset;
  if (tid >= n) return;
  int h = 0;
  int prev = offset;
  for (int i = 0; i < seg_size; ++i) {
    if (tid + i < n) {
      if (M[tid + i] == -1) {
        h = 0;
        prev = offset;
      } else {
        int k = M[tid + i];
        if (k != prev + 1 || !h) {
          while (tid + i + h < n && k + h < offset &&
                 t[tid + i + h] == t[k + h]) {
            ++h;
          }
        }
        prev = k;
      }
      lcp[tid + i] = h;
      if (h) {
        --h;
      }
    }
  }
}

void compute_LCP(thrust::device_vector<uint8_t>& T,
                 thrust::device_vector<int>& M, thrust::device_vector<int>& lcp,
                 size_t n, size_t offset) {
  uint8_t* t_ptr = thrust::raw_pointer_cast(T.data());
  int* M_ptr = thrust::raw_pointer_cast(M.data());
  int* lcp_ptr = thrust::raw_pointer_cast(lcp.data());
  uint32_t seg_size = divup(n - offset, SEGSIZE_FACTOR);
  uint32_t block_size = WARP_SIZE;
  uint32_t bl_num = divup(n - offset, block_size * seg_size);

  compute_PLCP_kernel<<<bl_num, block_size>>>(t_ptr, M_ptr, lcp_ptr, n,
                                              seg_size, offset);
  CHECK(cudaGetLastError());
  cudaDeviceSynchronize();
}

// M_l contains the positions in R and and M_r contains lengths of factors
__global__ void compute_MS_directly_kernel(uint8_t* t, int* M_l, int* M_r,
                                           size_t n, size_t seg_size,
                                           size_t offset) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  tid *= seg_size;
  tid += offset;
  if (tid >= n) return;
  int h_l = 0;
  int h_r = 0;
  int k_l = offset;
  int k_r = offset;
  int l_prev = offset;
  int r_prev = offset;
  for (int i = 0; i < seg_size; ++i) {
    if (tid + i < n) {
      if (M_l[tid + i] == -1) {
        h_l = 0;
        k_l = offset;
      } else {
        k_l = M_l[tid + i];
        if (k_l != l_prev + 1 || !h_l) {
          while (tid + i + h_l < n && k_l + h_l < offset &&
                 t[tid + i + h_l] == t[k_l + h_l]) {
            ++h_l;
          }
        }
        l_prev = k_l;
      }
      if (M_r[tid + i] == -1) {
        h_r = 0;
        k_r = offset;
      } else {
        k_r = M_r[tid + i];
        if (k_r != r_prev + 1) {
          while (tid + i + h_r < n && k_r + h_r < offset &&
                 t[tid + i + h_r] == t[k_r + h_r]) {
            ++h_r;
          }
        }
        r_prev = k_r;
      }
      if (h_l >= h_r) {
        M_l[tid + i] = k_l;
        M_r[tid + i] = h_l;
      } else {
        M_l[tid + i] = k_r;
        M_r[tid + i] = h_r;
      }
      if (h_l) {
        --h_l;
      }
      if (h_r) {
        --h_r;
      }
    }
  }
}

void compute_LCP_from_both(thrust::device_vector<uint8_t>& T,
                           thrust::device_vector<int>& M_l,
                           thrust::device_vector<int>& M_r, size_t n,
                           size_t offset) {
  uint8_t* t_ptr = thrust::raw_pointer_cast(T.data());
  int* M_l_ptr = thrust::raw_pointer_cast(M_l.data());
  int* M_r_ptr = thrust::raw_pointer_cast(M_r.data());
  uint32_t seg_size = divup(n - offset, SEGSIZE_FACTOR);
  uint32_t block_size = WARP_SIZE;
  uint32_t bl_num = divup(n - offset, block_size * seg_size);
  compute_MS_directly_kernel<<<bl_num, block_size>>>(t_ptr, M_l_ptr, M_r_ptr, n,
                                                     seg_size, offset);
  CHECK(cudaGetLastError());
  cudaDeviceSynchronize();
}

__global__ void compute_MS_directly_from_one_array_kernel(
    uint8_t* t, int* M_vector, size_t n, size_t seg_size, size_t offset) {
  size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  tid *= seg_size;
  tid += offset;
  if (tid >= n) return;
  int h_l = 0;
  int h_r = 0;
  int k_l = offset;
  int k_r = offset;
  int l_prev = offset;
  int r_prev = offset;
  for (int i = 0; i < seg_size; ++i) {
    bool a = true;
    bool b = true;
    if (tid + i < n) {
      if (M_vector[2 * (tid + i)] == -1) {
        h_l = 0;
        k_l = offset;
        a = false;
      } else {
        k_l = M_vector[2 * (tid + i)];
      }
      if (M_vector[2 * (tid + i) + 1] == -1) {
        h_r = 0;
        k_r = offset;
        b = false;
      } else {
        k_r = M_vector[2 * (tid + i) + 1];
      }
      if (a) {
        if (k_l != l_prev + 1 || !h_l) {
          while (tid + i + h_l < n && k_l + h_l < offset &&
                 t[tid + i + h_l] == t[k_l + h_l]) {
            ++h_l;
          }
        }
        l_prev = k_l;
      }
      if (b) {
        if (k_r != r_prev + 1) {
          while (tid + i + h_r < n && k_r + h_r < offset &&
                 t[tid + i + h_r] == t[k_r + h_r]) {
            ++h_r;
          }
        }
        r_prev = k_r;
      }
      if (h_l >= h_r) {
        M_vector[2 * (tid + i)] = k_l;
        M_vector[2 * (tid + i) + 1] = h_l;
      } else {
        M_vector[2 * (tid + i)] = k_r;
        M_vector[2 * (tid + i) + 1] = h_r;
      }
      if (h_l) {
        --h_l;
      }
      if (h_r) {
        --h_r;
      }
    }
  }
}

void compute_LCP_from_one_array_directly(thrust::device_vector<uint8_t>& T,
                                         thrust::device_vector<int>& M_vector,
                                         size_t n, size_t offset) {
  uint8_t* t_ptr = thrust::raw_pointer_cast(T.data());
  int* M_vector_ptr = thrust::raw_pointer_cast(M_vector.data());

  uint32_t seg_size = divup(n - offset, SEGSIZE_FACTOR);
  uint32_t block_size = WARP_SIZE;
  uint32_t bl_num = divup(n - offset, block_size * seg_size);

  compute_MS_directly_from_one_array_kernel<<<bl_num, block_size>>>(
      t_ptr, M_vector_ptr, n, seg_size, offset);
  CHECK(cudaGetLastError());
  cudaDeviceSynchronize();
}

void get_MS_values(thrust::device_vector<int>& l_M,
                   thrust::device_vector<int>& r_M,
                   thrust::device_vector<int>& l_lcp,
                   thrust::device_vector<int>& r_lcp, int n, int num_docs) {
  int* l_M_ptr = thrust::raw_pointer_cast(l_M.data());
  int* r_M_ptr = thrust::raw_pointer_cast(r_M.data());
  int* l_lcp_ptr = thrust::raw_pointer_cast(l_lcp.data());
  int* r_lcp_ptr = thrust::raw_pointer_cast(r_lcp.data());
  auto r = thrust::counting_iterator<int>(0);
  thrust::for_each(r, r + n, [=] __device__(int i) {
    if (l_lcp_ptr[i] >= r_lcp_ptr[i]) {
      r_lcp_ptr[i] = l_lcp_ptr[i];
      l_lcp_ptr[i] = l_M_ptr[i];
    } else {
      l_lcp_ptr[i] = r_M_ptr[i];
    }
  });
}

void mark_CMS_heads(thrust::device_vector<int>& lengths,
                    thrust::device_vector<int>& flags, int n, int offset) {
  int* len_ptr = thrust::raw_pointer_cast(lengths.data());
  int* flags_ptr = thrust::raw_pointer_cast(flags.data());
  auto r = thrust::counting_iterator<int>(0);
  thrust::for_each(r, r + n, [=] __device__(int i) {
    flags_ptr[i] = 0;
    if (i > offset && len_ptr[i] > len_ptr[i - 1] - 1) {
      flags_ptr[i] = 1;
    }

    if (i == offset) {
      flags_ptr[i] = 1;  // first element is always a head
    }
  });
}

void write_CMS_values(thrust::device_vector<int>& vec,
                      thrust::device_vector<int>& flags,
                      thrust::device_vector<int>& ranks,
                      thrust::device_vector<int>& pos,
                      thrust::device_vector<int>& len, int n) {
  int* vec_ptr = thrust::raw_pointer_cast(vec.data());
  int* flags_ptr = thrust::raw_pointer_cast(flags.data());
  int* ranks_ptr = thrust::raw_pointer_cast(ranks.data());
  int* pos_ptr = thrust::raw_pointer_cast(pos.data());
  int* len_ptr = thrust::raw_pointer_cast(len.data());
  auto r = thrust::counting_iterator<int>(0);
  thrust::for_each(r, r + n, [=] __device__(int i) {
    if (flags_ptr[i] == 1) {
      vec_ptr[(size_t)3 * ranks_ptr[i]] = pos_ptr[i];
      vec_ptr[(size_t)3 * ranks_ptr[i] + 1] = len_ptr[i];
      vec_ptr[(size_t)3 * ranks_ptr[i] + 2] = i;
    }
  });
}

void mark_CMS_heads_from_MS(thrust::device_vector<int>& MS,
                            thrust::device_vector<int>& flags, int n) {
  int* MS_ptr = thrust::raw_pointer_cast(MS.data());
  int* flags_ptr = thrust::raw_pointer_cast(flags.data());
  auto r = thrust::counting_iterator<int>(0);
  thrust::for_each(r, r + n, [=] __device__(int i) {
    int len = MS_ptr[(size_t)2 * i + 1];
    int len_prev = i ? MS_ptr[(size_t)2 * (i - 1) + 1] : 0;
    flags_ptr[i] = 0;
    if (i && len > len_prev - 1) {
      flags_ptr[i] = 1;
    }
    if (!i) {
      flags_ptr[i] = 1;  // first element is always a head
    }
  });
}

void write_CMS_values_from_MS(thrust::device_vector<int>& vec,
                              thrust::device_vector<int>& flags,
                              thrust::device_vector<int>& ranks,
                              thrust::device_vector<int>& MS, int n) {
  int* vec_ptr = thrust::raw_pointer_cast(vec.data());
  int* flags_ptr = thrust::raw_pointer_cast(flags.data());
  int* ranks_ptr = thrust::raw_pointer_cast(ranks.data());
  int* MS_ptr = thrust::raw_pointer_cast(MS.data());
  thrust::device_vector<size_t> heads_lens(vec.size() / 3, 0);
  size_t* heads_lens_ptr = thrust::raw_pointer_cast(heads_lens.data());
  auto r = thrust::counting_iterator<int>(0);
  thrust::for_each(r, r + n, [=] __device__(int i) {
    if (flags_ptr[i] == 1) {
      vec_ptr[(size_t)3 * ranks_ptr[i]] = MS_ptr[(size_t)i * 2];
      vec_ptr[(size_t)3 * ranks_ptr[i] + 1] = MS_ptr[(size_t)i * 2 + 1];
      vec_ptr[(size_t)3 * ranks_ptr[i] + 2] = i;
      heads_lens_ptr[ranks_ptr[i]] =
          (size_t)MS_ptr[(size_t)i * 2 + 1];
    }
  });
  size_t sum = thrust::reduce(heads_lens.begin(), heads_lens.end(), 0);
  std::cout << "Total length of heads: " << sum
            << ", number of heads: " << heads_lens.size()
            << ", length of S: " << n
            << ", sum(heads)/|S| ratio: " << (float)sum / n 
            << ", |S|/sum(heads) ratio: " << n/(float)sum 
            << std::endl;
}
