#pragma once

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cub/cub.cuh>
#include <fstream>
#include <iostream>
#include <vector>

void write_LRF(thrust::device_vector<uint32_t> &sa,
               thrust::device_vector<int> &lrf, thrust::device_vector<int> &lcp,
               size_t n) {
  uint32_t *sa_ptr = thrust::raw_pointer_cast(sa.data());
  int *lcp_ptr = thrust::raw_pointer_cast(lcp.data());
  int *lrf_ptr = thrust::raw_pointer_cast(lrf.data());
  auto r = thrust::counting_iterator<int>(0);
  thrust::for_each(r, r + n, [=] __device__(int i) {
    if (i == n - 1) {
      lrf_ptr[sa_ptr[i]] = lcp_ptr[i];
    } else {
      lrf_ptr[sa_ptr[i]] = max(lcp_ptr[i], lcp_ptr[i + 1]);
    }
  });
}

// Returns the leftmost occurrence of the element if it is present or (if not
// present) then returns -(x+1) where x is the index at which the key would be
// inserted into the array: i.e., the index of the first element greater than
// the key, or hi+1 if all elements in the array are less than the specified
// key.
int32_t __device__ binarySearchLB(int32_t lo, int32_t hi, uint32_t offset,
                                  uint8_t c, uint32_t *sa, uint8_t *ref) {
  uint32_t low = lo, high = hi;
  while (low <= high) {
    uint32_t mid = (low + high) >> 1;
    uint8_t midVal = ref[sa[mid] + offset];
    if (midVal < c) {
      low = mid + 1;
    } else if (midVal > c) {
      high = mid - 1;
    } else {                      // midVal == c
      if (mid == lo) return mid;  // leftmost occ of key found
      uint8_t midValLeft = ref[sa[mid - 1] + offset];
      if (midValLeft == midVal) {
        high = mid - 1;  // discard mid and the ones to the right of mid
      } else {           // midValLeft must be less than midVal == c
        return mid;      // leftmost occ of key found
      }
    }
  }
  return -(low + 1);  // key not found.
}

int32_t __device__ binarySearchRB(int32_t lo, int32_t hi, uint32_t offset,
                                  uint8_t c, uint32_t *sa, uint8_t *ref) {
  uint32_t low = lo, high = hi;
  while (low <= high) {
    uint32_t mid = (low + high) >> 1;
    uint8_t midVal = ref[sa[mid] + offset];
    if (midVal < c) {
      low = mid + 1;
    } else if (midVal > c) {
      high = mid - 1;
    } else {                      // midVal == c
      if (mid == hi) return mid;  // rightmost occ of key found
      uint8_t midValRight = ref[sa[mid + 1] + offset];
      if (midValRight == midVal) {
        low = mid + 1;  // discard mid and the ones to the left of mid
      } else {          // midValRight must be greater than midVal == c
        return mid;     // rightmost occ of key found
      }
    }
  }
  return -(low + 1);  // key not found.
}

void __device__ computeMatchingFactor(uint8_t *collection, uint8_t *reference,
                                      uint32_t *sa, uint64_t i, int32_t *pos,
                                      int32_t *len, int32_t &leftB,
                                      int32_t &rightB, int col_sz) {
  uint32_t offset = *len;
  uint64_t j = i + offset;

  int32_t nlb = leftB, nrb = rightB, maxMatch;
  uint32_t match = sa[nlb];
  while (j < col_sz) {  // scans the string from j onwards until a
    // maximal prefix is
    // found between reference and collection
    if (nlb == nrb) {
      if (reference[sa[nlb] + offset] != collection[j]) {
        break;
      }
      leftB = nlb;
      rightB = nrb;
      maxMatch = nlb;
    } else {  // refining the bucket in which the match is found, from left and
      // then from right
      nlb = binarySearchLB(nlb, nrb, offset, collection[j], sa, reference);
      if (nlb < 0) {
        // auto tmp = true;
        maxMatch = -nlb - 1;
        if (maxMatch == nrb + 1) {
          maxMatch--;
        }
        match = sa[maxMatch];
        break;
      }
      nrb = binarySearchRB(nlb, nrb, offset, collection[j], sa, reference);

      leftB = nlb;
      rightB = nrb;
    }
    match = sa[nlb];
    j++;
    offset++;
  }
  *pos = match;
  *len = offset;
}

int32_t __device__ psv(int *A, int32_t i, uint32_t x) {
  int32_t ni = i;
  while (ni >= 0) {
    if (A[ni] < x) return ni;
    --ni;
  }
  return 0;
}

int32_t __device__ nsv(int *A, int32_t i, uint32_t x, uint32_t n) {
  int32_t ni = i + 1;
  while (ni < n) {
    if (A[ni] < x) return ni - 1;
    ++ni;
  }
  return n - 1;
}

void __device__ contract_left(int *lcp, uint32_t *sa, uint32_t *isa,
                              int32_t &lo, int32_t &hi, uint len, uint32_t n) {
  uint32_t lb = sa[lo];
  uint32_t rb = sa[hi];
  if (lb == n - 1 || rb == n - 1) {
    lo = 0;
    hi = n - 1;
    return;
  }
  auto temp_l = isa[lb + 1];
  auto temp_r = isa[rb + 1];
  lb = psv(lcp, temp_l, len);
  rb = nsv(lcp, temp_r, len, n);
  lo = lb;
  hi = rb;
}

void computeMS_thrust(thrust::device_vector<uint8_t> &R,
               thrust::device_vector<uint8_t> &S,
               thrust::device_vector<uint> &SA_R,
               thrust::device_vector<uint> &ISA_R,
               thrust::device_vector<int> &LCP_R,
               thrust::device_vector<int> &LRF_R,
               thrust::device_vector<int> &MS, int seg_size, bool lrf_heuristic,
               int lcp_max = 0) {
  uint *SA_R_ptr = thrust::raw_pointer_cast(SA_R.data());
  uint *ISA_R_ptr = thrust::raw_pointer_cast(ISA_R.data());
  int *LCP_R_ptr = thrust::raw_pointer_cast(LCP_R.data());
  int *LRF_ptr = thrust::raw_pointer_cast(LRF_R.data());
  uint8_t *R_ptr = thrust::raw_pointer_cast(R.data());
  uint8_t *S_ptr = thrust::raw_pointer_cast(S.data());
  int32_t ref_sz = R.size();
  int32_t n_s = S.size();
  int *MS_ptr = thrust::raw_pointer_cast(MS.data());
  int thread_num = divup(n_s, seg_size);
  
  auto r = thrust::counting_iterator<int>(0);
  thrust::for_each(r, r + thread_num, [=] __device__(int i) {
    uint32_t start = i * seg_size;
    uint32_t thr_end = min((i + 1) * seg_size, n_s);
    if (start >= n_s) return;

    int32_t leftB = 0;
    int32_t rightB = ref_sz - 1;
    int32_t pos = ref_sz - 1, len = 0;

    while (start < thr_end && start < n_s) {
      computeMatchingFactor(S_ptr, R_ptr, SA_R_ptr, start, &pos, &len, leftB,
                            rightB, n_s);
      MS_ptr[start * 2] = pos;
      MS_ptr[start * 2 + 1] = len;

      if (len) len--;
      if (start + 1 < n_s && start + 1 < thr_end) {
        if ((lrf_heuristic || lcp_max) && leftB == rightB) {
          if (lrf_heuristic) {
            while (len > LRF_ptr[pos + 1] && start < n_s && start < thr_end) {
              start++;
              pos++;
              MS_ptr[start * 2] = pos;
              MS_ptr[start * 2 + 1] = len;
              len--;
            }
          } else {
            while (len > lcp_max && start < n_s && start < thr_end) {
              start++;
              pos++;
              MS_ptr[start * 2] = pos;
              MS_ptr[start * 2 + 1] = len;
              len--;
            }
          }
        }
        if (start + 1 < n_s && start + 1 < thr_end) {
          contract_left(LCP_R_ptr, SA_R_ptr, ISA_R_ptr, leftB, rightB, len,
                        ref_sz);
        }
      }

      start++;
    }
  });
}

__global__ void compute_MS(uint8_t *collection, uint32_t *col_offs,
                           uint8_t *reference, uint32_t *sa, uint32_t *isa,
                           int *lcp, int *lrf, int32_t *MS, int32_t ref_sz,
                           bool lrf_heuristic, int blocks, size_t seg_sz,
                           int lcp_max = 0) {
  uint32_t tid = threadIdx.x;
  uint32_t bid = blockIdx.x % blocks;
  uint32_t col_st =
      blockIdx.x / blocks ? col_offs[blockIdx.x / blocks - 1] + 1 : 0;
  uint32_t col_end = col_offs[blockIdx.x / blocks];
  /* uint32_t seg_sz =
      ((col_end - col_st) + (blockDim.x * blocks) - 1) / (blockDim.x *
     blocks);
   */
  uint32_t start = col_st + tid * seg_sz + bid * seg_sz * blockDim.x;
  uint32_t s_end = start + seg_sz;
  int32_t leftB = 0;
  int32_t rightB = ref_sz - 1;
  int32_t pos = ref_sz - 1, len = 0;
  if (start >= col_end) return;
  while (start < s_end && start < col_end) {
    computeMatchingFactor(collection, reference, sa, start, &pos, &len, leftB,
                          rightB, col_end);
    MS[start * 2] = pos;
    MS[start * 2 + 1] = len;

    if (len) len--;
    if (start + 1 < s_end && start + 1 < col_end) {
      if ((lrf_heuristic || lcp_max) && leftB == rightB) {
        if (lrf_heuristic) {
          while (len > lrf[pos + 1] && start < s_end && start < col_end) {
            start++;
            pos++;
            MS[start * 2] = pos;
            MS[start * 2 + 1] = len;
            len--;
          }
        } else {
          while (len > lcp_max && start < s_end && start < col_end) {
            start++;
            pos++;
            MS[start * 2] = pos;
            MS[start * 2 + 1] = len;
            len--;
          }
        }
      }
      if (start + 1 < s_end && start + 1 < col_end) {
        contract_left(lcp, sa, isa, leftB, rightB, len, ref_sz);
      }
    }
    start++;
  }
  if (!tid && !bid) {
    MS[(col_end) * 2] = ref_sz - 1;
    MS[(col_end) * 2 + 1] = 0;
  }
}
