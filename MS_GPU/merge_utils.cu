#pragma once

#include "utils.cu"

// Returns the leftmost occurrence of the element if it is present or (if not
// present) then returns -(x+1) where x is the index at which the key would be
// inserted into the array: i.e., the index of the first element greater than
// the key, or hi+1 if all elements in the array are less than the specified
// key.
int32_t __device__ binarySearchLB(int32_t lo, int32_t hi, uint32_t offset,
                                  uint8_t c, uint* sa_r, uint8_t* ref) {
  int32_t low = lo, high = hi;
  while (low <= high) {
    int32_t mid = (low + high) >> 1;
    uint8_t midVal = ref[sa_r[mid] + offset];
    if (midVal < c) {
      low = mid + 1;
    } else if (midVal > c) {
      high = mid - 1;
    } else {                      // midVal == c
      if (mid == lo) return mid;  // leftmost occ of key found
      uint8_t midValLeft = ref[sa_r[mid - 1] + offset];
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
                                  uint8_t c, uint* sa_r, uint8_t* ref) {
  int32_t low = lo, high = hi;
  while (low <= high) {
    int32_t mid = (low + high) >> 1;
    uint8_t midVal = ref[sa_r[mid] + offset];
    if (midVal < c) {
      low = mid + 1;
    } else if (midVal > c) {
      high = mid - 1;
    } else {                      // midVal == c
      if (mid == hi) return mid;  // rightmost occ of key found
      uint8_t midValRight = ref[sa_r[mid + 1] + offset];
      if (midValRight == midVal) {
        low = mid + 1;  // discard mid and the ones to the left of mid
      } else {          // midValRight must be greater than midVal == c
        return mid;     // rightmost occ of key found
      }
    }
  }
  return -(low + 1);  // key not found.
}

void __device__ computeMatchingFactor(uint8_t* s, uint8_t* r, uint* sa_r,
                                      uint64_t i_s, int32_t* pos, int32_t* len,
                                      int s_sz, int r_sz) {
  uint32_t offset = 0;
  uint64_t j = i_s + offset;

  int32_t nlb = 0, nrb = r_sz - 1, maxMatch;
  unsigned int match = nlb;
  while (j < s_sz) {
    // scans the string from j onwards until a
    // maximal prefix is found between r and s
    if (nlb == nrb) {
      if (r[sa_r[nlb] + offset] != s[j]) {
        break;
      }
      maxMatch = nlb;
    } else {
      // refining the bucket in which the match is found,
      // from left and then from right
      nlb = binarySearchLB(nlb, nrb, offset, s[j], sa_r, r);
      if (nlb < 0) {
        // auto tmp = true;
        maxMatch = -nlb - 1;
        if (maxMatch == nrb + 1) {
          maxMatch--;
        }
        match = maxMatch;
        break;
      }
      nrb = binarySearchRB(nlb, nrb, offset, s[j], sa_r, r);
    }
    match = nlb;
    j++;
    offset++;
  }
  *pos = match;
  *len = offset;
}

void get_offsets_and_lens(thrust::device_vector<uint>& SA_R,
                          thrust::device_vector<uint>& SA_S,
                          thrust::device_vector<uint8_t> R,
                          thrust::device_vector<uint8_t> S,
                          thrust::device_vector<int>& offsets_and_lens, int n_r,
                          int n_s, int chunck_size/* ,
                          thrust::device_vector<int>& flags_l */) {
  uint* SA_R_ptr = thrust::raw_pointer_cast(SA_R.data());
  uint* SA_S_ptr = thrust::raw_pointer_cast(SA_S.data());
  uint8_t* R_ptr = thrust::raw_pointer_cast(R.data());
  uint8_t* S_ptr = thrust::raw_pointer_cast(S.data());
  int* offsets_ptr = thrust::raw_pointer_cast(offsets_and_lens.data());
  /*  int* flags_ptr = thrust::raw_pointer_cast(flags_l.data()); */

  int thread_nums = divup(n_s, chunck_size);
  auto r = thrust::counting_iterator<int>(0);

  thrust::for_each(r, r + thread_nums, [=] __device__(int i) {
    if (i * chunck_size >= n_s) return;
    uint s_rank = i * chunck_size;
    uint s_offset = SA_S_ptr[s_rank];
    int left_bracket = 0;
    int lcp_l = 0;
    computeMatchingFactor(S_ptr, R_ptr, SA_R_ptr, s_offset, &left_bracket,
                          &lcp_l, n_s, n_r);
    if (R_ptr[SA_R_ptr[left_bracket] + lcp_l] > S_ptr[s_offset + lcp_l]) {
      left_bracket--;
      lcp_l = 0;
    }
    offsets_ptr[i * 2] = left_bracket;
    offsets_ptr[i * 2 + 1] = lcp_l;
    /* printf("i: %d, s_rank: %d, s_offset: %d, left_bracket: %d, lcp_l: %d\n",
       i, s_rank, s_offset, left_bracket, lcp_l); */
    /* int actual_l_bracket = flags_ptr[s_offset + n_r];
    if (actual_l_bracket != SA_R_ptr[left_bracket]) {
      if (actual_l_bracket == SA_R_ptr[left_bracket - 1]) {
        printf("CHrist!!\n");
      } else {
        printf(
            "i: %d, s_rank: %d, s_offset: %d, left_bracket: %d, lcp_l: %d, "
            "actual _l %d\n",
            i, s_rank, s_offset, SA_R_ptr[left_bracket], lcp_l,
            actual_l_bracket);
      }
    } else {
      printf(
          "Correct on i: %d, s_rank: %d, s_offset: %d, left_bracket: %d, "
          "lcp_l: %d, actual _l: %d\n",
          i, s_rank, s_offset, SA_R_ptr[left_bracket], lcp_l, actual_l_bracket);
    } */
  });
}

int32_t __device__ lcp(uint32_t i, uint32_t j, uint8_t* r, uint8_t* s, int n_r,
                       int n_s) {
  int l = 0;
  while (i + l < n_r && j + l < n_s && r[i + l] == s[j + l]) {
    l++;
  }
  return l;
}

// Compute the left brackets of S with respect to SA_R, the right brackets
// will be the concecutive elements of the left brackets in SA_R
// The left brackets will be stored in S_predesessors in text order
void get_brackets_GPU(
    thrust::device_vector<uint8_t>& R, thrust::device_vector<uint8_t>& S,
    thrust::device_vector<int>& SA_R, thrust::device_vector<int>& SA_S,
    thrust::device_vector<int>& LCP_R, thrust::device_vector<int>& LCP_S,
    thrust::device_vector<int>& offsets_and_lens,
    thrust::device_vector<int>& S_predesessors, int chunck_size) {
  int* SA_R_ptr = thrust::raw_pointer_cast(SA_R.data());
  int* SA_S_ptr = thrust::raw_pointer_cast(SA_S.data());
  int* LCP_R_ptr = thrust::raw_pointer_cast(LCP_R.data());
  int* LCP_S_ptr = thrust::raw_pointer_cast(LCP_S.data());
  uint8_t* R_ptr = thrust::raw_pointer_cast(R.data());
  uint8_t* S_ptr = thrust::raw_pointer_cast(S.data());
  int* offsets_ptr = thrust::raw_pointer_cast(offsets_and_lens.data());
  int* S_predesessors_ptr = thrust::raw_pointer_cast(S_predesessors.data());
  int n_r = R.size();
  int n_s = S.size();

  int thread_nums = divup(n_s, chunck_size);
  auto r = thrust::counting_iterator<int>(0);
  printf("thread_nums: %d, n_s: %d, chunck_size: %d\n", thread_nums, n_s,
         chunck_size);

  thrust::for_each(r, r + thread_nums, [=] __device__(int i) {
    if (i * chunck_size >= n_s) return;
    int s_offset = i * chunck_size;
    int left_bracket = offsets_ptr[i * 2];
    int lcp_l = offsets_ptr[i * 2 + 1];
    /* printf("s_offset: %d, left_bracket: %d, lcp_l: %d\n", s_offset,
           left_bracket, lcp_l); */

    int i_s = s_offset;
    int i_r = left_bracket;
    int suff_S = SA_S_ptr[i_s];
    int suff_R = SA_R_ptr[i_r];
    int prev_i_r = i_r;
    int l = lcp_l;
    l = l + lcp(suff_R + l, suff_S + l, R_ptr, S_ptr, n_r, n_s);
    while (i_s < min(n_s, s_offset + chunck_size) && i_r < n_r) {
      suff_S = SA_S_ptr[i_s];
      suff_R = SA_R_ptr[i_r];
      while (S_ptr[suff_S + l] > R_ptr[suff_R + l] && i_r < n_r) {
        prev_i_r = i_r;
        i_r++;
        if (i_r < n_r) {
          suff_R = SA_R_ptr[i_r];
          l = min(l, LCP_R_ptr[i_r]);
          l = l + lcp(suff_R + l, suff_S + l, R_ptr, S_ptr, n_r, n_s);
        } else {
          break;
        }
      }
      /* printf("after first loop i_s: %d, i_r: %d, prev_i_r: %d, l: %d\n", i_s,
       * i_r, prev_i_r, l);
       */
      while (S_ptr[suff_S + l] < R_ptr[suff_R + l] &&
             i_s < min(n_s, s_offset + chunck_size)) {
        S_predesessors_ptr[suff_S] = SA_R_ptr[prev_i_r];  // the left bracket
        ++i_s;
        if (i_s < min(n_s, s_offset + chunck_size)) {
          suff_S = SA_S_ptr[i_s];
          l = min(l, LCP_S_ptr[i_s]);
          l = l + lcp(suff_R + l, suff_S + l, R_ptr, S_ptr, n_r, n_s);
        } else {
          break;
        }
      }
 /*    printf("after second loop i_s: %d, i_r: %d, prev_i_r: %d, l: %d\n", i_s, i_r, prev_i_r, l);
  */ }
 while (i_s < min(n_s, s_offset + chunck_size)) {
   S_predesessors_ptr[SA_S_ptr[i_s]] = SA_R_ptr[prev_i_r];
   ++i_s;
   // printf("i_s: %d, i_r: %d, prev_i_r: %d, l: %d\n", i_s, i_r, prev_i_r,
   // l);
 }
  });
  CHECK(cudaGetLastError());
  cudaDeviceSynchronize();
}

// Compute the left brackets of S with respect to SA_R, the right brackets
// will be the concecutive elements of the left brackets in SA_R
// The left brackets will be stored in S_predesessors in text order
void get_brackets_GPU_smarter(
    thrust::device_vector<uint8_t>& R, thrust::device_vector<uint8_t>& S,
    thrust::device_vector<int>& SA_R, thrust::device_vector<int>& SA_S,
    thrust::device_vector<int>& LCP_R, thrust::device_vector<int>& LCP_S,
    thrust::device_vector<int>& offsets_and_lens,
    thrust::device_vector<int>& S_predesessors, int chunck_size) {
  int* SA_R_ptr = thrust::raw_pointer_cast(SA_R.data());
  int* SA_S_ptr = thrust::raw_pointer_cast(SA_S.data());
  int* LCP_R_ptr = thrust::raw_pointer_cast(LCP_R.data());
  int* LCP_S_ptr = thrust::raw_pointer_cast(LCP_S.data());
  uint8_t* R_ptr = thrust::raw_pointer_cast(R.data());
  uint8_t* S_ptr = thrust::raw_pointer_cast(S.data());
  int* offsets_ptr = thrust::raw_pointer_cast(offsets_and_lens.data());
  int* S_predesessors_ptr = thrust::raw_pointer_cast(S_predesessors.data());
  int n_r = R.size();
  int n_s = S.size();

  int thread_nums = divup(n_s, chunck_size);
  auto r = thrust::counting_iterator<int>(0);
  printf("thread_nums: %d, n_s: %d, chunck_size: %d\n", thread_nums, n_s,
         chunck_size);

  thrust::for_each(r, r + thread_nums, [=] __device__(int i) {
    if (i * chunck_size >= n_s) return;
    int s_offset = i * chunck_size;
    int left_bracket = offsets_ptr[i * 2];
    int lcp_l = offsets_ptr[i * 2 + 1];
    /* printf("s_offset: %d, left_bracket: %d, lcp_l: %d\n", s_offset,
           left_bracket, lcp_l); */
    int i_s = s_offset;
    int i_r = left_bracket;
    int suff_S = SA_S_ptr[i_s];
    int suff_R = SA_R_ptr[i_r];
    int prev_i_r = i_r;
    int l = lcp_l;
    l = l + lcp(suff_R + l, suff_S + l, R_ptr, S_ptr, n_r, n_s);
    while (i_s < min(n_s, s_offset + chunck_size) && i_r < n_r) {
      suff_S = SA_S_ptr[i_s];
      suff_R = SA_R_ptr[i_r];
      while (S_ptr[suff_S + l] > R_ptr[suff_R + l] && i_r < n_r) {
        prev_i_r = i_r;
        i_r++;
        if (i_r < n_r) {
          suff_R = SA_R_ptr[i_r];
          int lcp_R = LCP_R_ptr[i_r];
          while (lcp_R > l) {
            prev_i_r = i_r;
            i_r++;
            if (i_r >= n_r) break;
            lcp_R = LCP_R_ptr[i_r];
          }
          if (lcp_R == l) {
            // this might be the right bracket
            suff_R = SA_R_ptr[i_r];
            l = l + lcp(suff_R + l, suff_S + l, R_ptr, S_ptr, n_r, n_s);
            break;
          } else {
            l = lcp_R;
            break;
          }
        } else {
          break;
        }
      }
      while (S_ptr[suff_S + l] < R_ptr[suff_R + l] &&
             i_s < min(n_s, s_offset + chunck_size)) {
        S_predesessors_ptr[suff_S] = SA_R_ptr[prev_i_r];  // the left bracket
        ++i_s;
        if (i_s < min(n_s, s_offset + chunck_size)) {
          suff_S = SA_S_ptr[i_s];
          int lcp_S = LCP_S_ptr[i_s];
          while (lcp_S > l) {
            S_predesessors_ptr[suff_S] = SA_R_ptr[prev_i_r];
            ++i_s;
            if (i_s >= min(n_s, s_offset + chunck_size)) break;
            lcp_S = LCP_S_ptr[i_s];
            suff_S = SA_S_ptr[i_s];
          }
          if (lcp_S == l) {
            suff_S = SA_S_ptr[i_s];
            l = l + lcp(suff_R + l, suff_S + l, R_ptr, S_ptr, n_r, n_s);
            break;
          } else {
            l = lcp_S;
            break;
          }
        } else {
          break;
        }
      }
    }
    while (i_s < min(n_s, s_offset + chunck_size)) {
      S_predesessors_ptr[SA_S_ptr[i_s]] = SA_R_ptr[prev_i_r];
      ++i_s;
    }
  });
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
}

// Compute the left brackets of S with respect to SA_R, the right brackets
// will be the concecutive elements of the left brackets in SA_R
// The left brackets will be stored in S_predesessors in text order
void get_factors_directly(thrust::device_vector<uint8_t>& R,
                          thrust::device_vector<uint8_t>& S,
                          thrust::device_vector<uint>& SA_R,
                          thrust::device_vector<uint>& SA_S,
                          thrust::device_vector<int>& LCP_R,
                          thrust::device_vector<int>& LCP_S,
                          thrust::device_vector<int>& offsets_and_lens,
                          thrust::device_vector<int>& MS, int chunck_size) {
  uint* SA_R_ptr = thrust::raw_pointer_cast(SA_R.data());
  uint* SA_S_ptr = thrust::raw_pointer_cast(SA_S.data());
  int* LCP_R_ptr = thrust::raw_pointer_cast(LCP_R.data());
  int* LCP_S_ptr = thrust::raw_pointer_cast(LCP_S.data());
  uint8_t* R_ptr = thrust::raw_pointer_cast(R.data());
  uint8_t* S_ptr = thrust::raw_pointer_cast(S.data());
  int* offsets_ptr = thrust::raw_pointer_cast(offsets_and_lens.data());

  int* MS_ptr = thrust::raw_pointer_cast(MS.data());
  int n_r = R.size();
  int n_s = S.size();

  int thread_nums = divup(n_s, chunck_size);
  auto r = thrust::counting_iterator<int>(0);
  /* printf("thread_nums: %d, n_s: %d, chunck_size: %d, n_r: %d\n", thread_nums,
         n_s, chunck_size, n_r);
 */
  thrust::for_each(r, r + thread_nums, [=] __device__(int i) {
    if (i * chunck_size >= n_s) return;
    int s_offset = i * chunck_size;
    int left_bracket = offsets_ptr[i * 2];
    int lcp_l = offsets_ptr[i * 2 + 1];
    int prev_lcp = 0;
    bool first = false;
    int i_s = s_offset;
    int i_r = left_bracket;
    size_t suff_S = SA_S_ptr[i_s];
    size_t suff_R = SA_R_ptr[i_r];
    int prev_i_r = i_r;
    int l = lcp_l;
    l = l + lcp(suff_R + l, suff_S + l, R_ptr, S_ptr, n_r, n_s);
    prev_lcp = l;

    while (i_s < min(n_s, s_offset + chunck_size) && i_r < n_r) {
      suff_S = SA_S_ptr[i_s];
      suff_R = SA_R_ptr[i_r];
      while (S_ptr[suff_S + l] > R_ptr[suff_R + l] && i_r < n_r) {
        first = true;
        prev_i_r = i_r;
        prev_lcp = l;
        i_r++;
        if (i_r >= n_r) break;
        suff_R = SA_R_ptr[i_r];
        int lcp_R = LCP_R_ptr[i_r];
        while (lcp_R > l) {
          prev_i_r = i_r;
          i_r++;
          if (i_r >= n_r) break;
          lcp_R = LCP_R_ptr[i_r];
        }
        if (i_r >= n_r) break;
        if (lcp_R == l) {
          // this might be the right bracket, if so, current l will be lcp of
          // the left bracket but the lcp with the right bracket will be at
          // least the lcp of left bracket, we could just drop it here
          suff_R = SA_R_ptr[i_r];
          l = l + lcp(suff_R + l, suff_S + l, R_ptr, S_ptr, n_r, n_s);
          break;
        } else if (lcp_R < l) {
          // suffix pointed to by i_r is lexicographically larger then s_i
          // suffix and is the right bracket, but the left bracket has a
          // larger lcp with the s_i suffix and thus is the matching factor.
          prev_lcp = l;
          l = lcp_R;
          break;
        }
      }

      while (S_ptr[suff_S + l] < R_ptr[suff_R + l] &&
             i_s < min(n_s, s_offset + chunck_size)) {
        first = false;
        if (prev_lcp >= l) {
          MS_ptr[suff_S * 2] =
              SA_R_ptr[prev_i_r];             // pos of the MS (left bracket)
          MS_ptr[suff_S * 2 + 1] = prev_lcp;  // len of the MS
        } else {
          MS_ptr[suff_S * 2] = suff_R;  // pos of the MS (right bracket)
          MS_ptr[suff_S * 2 + 1] = l;   // len of the MS
        }
        ++i_s;
        if (i_s >= min(n_s, s_offset + chunck_size)) break;
        suff_S = SA_S_ptr[i_s];
        int lcp_S = LCP_S_ptr[i_s];
        prev_lcp = min(lcp_S, prev_lcp);
        while (lcp_S > l) {
          if (prev_lcp >= l) {
            MS_ptr[suff_S * 2] = SA_R_ptr[prev_i_r];  // pos of the MS
            MS_ptr[suff_S * 2 + 1] = prev_lcp;        // len of the MS
          } else {
            MS_ptr[suff_S * 2] = suff_R;  // pos of the MS (right bracket)
            MS_ptr[suff_S * 2 + 1] = l;   // len of the MS
          }
          ++i_s;
          if (i_s >= min(n_s, s_offset + chunck_size)) break;
          lcp_S = LCP_S_ptr[i_s];
          suff_S = SA_S_ptr[i_s];
          prev_lcp = min(lcp_S, prev_lcp);
        }
        if (i_s >= min(n_s, s_offset + chunck_size)) break;
        if (lcp_S == l) {
          suff_S = SA_S_ptr[i_s];
          l = l + lcp(suff_R + l, suff_S + l, R_ptr, S_ptr, n_r, n_s);
          break;
        } else if (lcp_S < l) {
          // suff_S belongs to another interval in SA_R
          l = lcp_S;
          prev_lcp = l;
          break;
        }
      }
    }
    while (i_s < min(n_s, s_offset + chunck_size)) {
      suff_S = SA_S_ptr[i_s];

      MS_ptr[suff_S * 2] = SA_R_ptr[prev_i_r];  // pos of the MS
      MS_ptr[suff_S * 2 + 1] =
          first ? l : min(l, LCP_S_ptr[i_s]);  // len of the MS
      if (first) first = false;
      ++i_s;
      if (i_s >= min(n_s, s_offset + chunck_size)) break;
      l = min(l, LCP_S_ptr[i_s]);
    }
  });
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
}

void get_matching_factor_offsets(thrust::device_vector<uint>& SA_R,
                                 thrust::device_vector<uint>& SA_S,
                                 thrust::device_vector<uint8_t> R,
                                 thrust::device_vector<uint8_t> S,
                                 thrust::device_vector<int>& offsets_and_lens,
                                 int n_r, int n_s, int chunck_size) {
  uint* SA_R_ptr = thrust::raw_pointer_cast(SA_R.data());
  uint* SA_S_ptr = thrust::raw_pointer_cast(SA_S.data());
  uint8_t* R_ptr = thrust::raw_pointer_cast(R.data());
  uint8_t* S_ptr = thrust::raw_pointer_cast(S.data());
  int* offsets_ptr = thrust::raw_pointer_cast(offsets_and_lens.data());

  int thread_nums = divup(n_s, chunck_size);
  auto r = thrust::counting_iterator<int>(0);

  thrust::for_each(r, r + thread_nums, [=] __device__(int i) {
    if (i * chunck_size >= n_s) return;
    uint s_rank = i * chunck_size;
    uint s_offset = SA_S_ptr[s_rank];
    int pos = n_r;
    int lcp_l = 0;
    computeMatchingFactor(S_ptr, R_ptr, SA_R_ptr, s_offset, &pos, &lcp_l, n_s,
                          n_r);

    offsets_ptr[i * 2] = pos;
    offsets_ptr[i * 2 + 1] = lcp_l;
  });
}

void get_factors_directly_circular_R(
    thrust::device_vector<uint8_t>& R, thrust::device_vector<uint8_t>& S,
    thrust::device_vector<uint>& SA_R, thrust::device_vector<uint>& SA_S,
    thrust::device_vector<int>& LCP_R, thrust::device_vector<int>& LCP_S,
    thrust::device_vector<int>& offsets_and_lens,
    thrust::device_vector<int>& MS, int chunck_size) {
  uint* SA_R_ptr = thrust::raw_pointer_cast(SA_R.data());
  uint* SA_S_ptr = thrust::raw_pointer_cast(SA_S.data());
  int* LCP_R_ptr = thrust::raw_pointer_cast(LCP_R.data());
  int* LCP_S_ptr = thrust::raw_pointer_cast(LCP_S.data());
  uint8_t* R_ptr = thrust::raw_pointer_cast(R.data());
  uint8_t* S_ptr = thrust::raw_pointer_cast(S.data());
  int* offsets_ptr = thrust::raw_pointer_cast(offsets_and_lens.data());

  int* MS_ptr = thrust::raw_pointer_cast(MS.data());
  int n_r = R.size();
  int n_s = S.size();

  int thread_nums = divup(n_s, chunck_size);
  thrust::device_vector<uint32_t> lcp_calls_vec(thread_nums, 0);
  thrust::device_vector<uint32_t> lcp_sums_vec(thread_nums, 0);
  uint32_t* lcp_calls_ptr = thrust::raw_pointer_cast(lcp_calls_vec.data());
  uint32_t* lcp_sums_ptr = thrust::raw_pointer_cast(lcp_sums_vec.data());
  auto r = thrust::counting_iterator<int>(0);
  /* printf("thread_nums: %d, n_s: %d, chunck_size: %d, n_r: %d\n", thread_nums,
  n_s, chunck_size, n_r);
  */
  thrust::for_each(r, r + thread_nums, [=] __device__(int i) {
    if (i * chunck_size >= n_s) return;
    int s_offset = i * chunck_size;
    int matching_factor = offsets_ptr[i * 2];
    int lcp_mf = offsets_ptr[i * 2 + 1];
    int prev_lcp = 0;
    bool first = false;
    int i_s = s_offset;
    int i_r = matching_factor;
    size_t suff_S;
    size_t suff_R;
    int prev_i_r = -1;
    int l = lcp_mf;
    // l = l + lcp(suff_R + l, suff_S + l, R_ptr, S_ptr, n_r, n_s);
    // prev_lcp = l;
    while (i_s < min(n_s, s_offset + chunck_size) /*  && i_r < n_r */) {
      suff_S = SA_S_ptr[i_s];
      suff_R = SA_R_ptr[i_r];
      while (S_ptr[suff_S + l] > R_ptr[suff_R + l] && i_r < n_r) {
        first = true;
        prev_i_r = i_r;
        prev_lcp = l;
        i_r++;
        if (i_r >= n_r) break;
        suff_R = SA_R_ptr[i_r];
        int lcp_R = LCP_R_ptr[i_r];
        while (lcp_R > l) {
          prev_i_r = i_r;
          i_r++;
          if (i_r >= n_r) break;
          lcp_R = LCP_R_ptr[i_r];
        }
        if (i_r >= n_r) break;
        suff_R = SA_R_ptr[i_r];
        if (lcp_R == l) {
          // this might be the right bracket, if so, current l will be lcp of
          // the left bracket but the lcp with the right bracket will be at
          // least the lcp of left bracket, we could just drop it here
          l = l + lcp(suff_R + l, suff_S + l, R_ptr, S_ptr, n_r, n_s);
          lcp_calls_ptr[i]++;
          lcp_sums_ptr[i] += l - lcp_R;
          break;
        } else if (lcp_R < l) {
          // suffix pointed to by i_r is lexicographically larger then s_i
          // suffix and is the right bracket, but the left bracket has a
          // larger lcp with the s_i suffix and thus is the matching factor.
          prev_lcp = l;
          l = lcp_R;
          break;
        }
      }
      while (S_ptr[suff_S + l] < R_ptr[suff_R + l] &&
             i_s < min(n_s, s_offset + chunck_size)) {
        first = false;
        if (prev_lcp > l && prev_i_r >= 0) {
          MS_ptr[suff_S * 2] = SA_R_ptr[prev_i_r];  // pos of the left bracket
          MS_ptr[suff_S * 2 + 1] = prev_lcp;        // len of the MS
        } else {
          MS_ptr[suff_S * 2] = suff_R;  // pos of the MS (right bracket)
          MS_ptr[suff_S * 2 + 1] = l;   // len of the MS
        }
        ++i_s;
        if (i_s >= min(n_s, s_offset + chunck_size)) break;
        suff_S = SA_S_ptr[i_s];
        int lcp_S = LCP_S_ptr[i_s];
        prev_lcp = min(lcp_S, prev_lcp);
        while (lcp_S > l) {
          if (prev_lcp > l && prev_i_r >= 0) {
            MS_ptr[suff_S * 2] = SA_R_ptr[prev_i_r];  // pos of the MS
            MS_ptr[suff_S * 2 + 1] = prev_lcp;        // len of the MS
          } else {
            MS_ptr[suff_S * 2] = suff_R;  // pos of the MS (right bracket)
            MS_ptr[suff_S * 2 + 1] = l;   // len of the MS
          }
          ++i_s;
          if (i_s >= min(n_s, s_offset + chunck_size)) break;
          lcp_S = LCP_S_ptr[i_s];
          suff_S = SA_S_ptr[i_s];
          prev_lcp = min(lcp_S, prev_lcp);
        }
        if (i_s >= min(n_s, s_offset + chunck_size)) break;
        if (!lcp_S) {
          // suff_S may be the start of a new document SA and we need to restart
          // the index in SA_R
          if (S_ptr[suff_S] == '$') {
            /*   printf("Restarting i_r to 0 from %d, prev_i_r to -1, l to 0\n",
                     i_r); */
            i_r = 0;
            prev_i_r = -1;
            prev_lcp = 0;  // this should be already done
            l = 0;
            break;
          }
        }
        if (lcp_S == l) {
          suff_S = SA_S_ptr[i_s];
          l = l + lcp(suff_R + l, suff_S + l, R_ptr, S_ptr, n_r, n_s);
          lcp_calls_ptr[i]++;
          lcp_sums_ptr[i] += l - lcp_S;
          break;
        }
        if (lcp_S < l) {
          // suff_S belongs to another interval in SA_R
          l = lcp_S;
          prev_lcp = l;
          break;
        }
      }
      while (i_r >= n_r && i_s < min(n_s, s_offset + chunck_size)) {
        suff_S = SA_S_ptr[i_s];
        if (!l) {
          // suff_S may be the start of a new document SA and we need to restart
          // the index in SA_R
          if (S_ptr[suff_S] == '$') {
            i_r = 0;
            prev_i_r = -1;
            prev_lcp = 0;  // this should be already done
            l = 0;
            break;
          }
        }

        MS_ptr[suff_S * 2] = SA_R_ptr[prev_i_r];  // pos of the MS
        MS_ptr[suff_S * 2 + 1] =
            first ? l : min(l, LCP_S_ptr[i_s]);  // len of the MS
        if (first) first = false;
        ++i_s;
        if (i_s >= min(n_s, s_offset + chunck_size)) break;
        l = min(l, LCP_S_ptr[i_s]);
      }
    }
  });
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  // sum the lcp calls and ans sums
  uint32_t lcp_calls = thrust::reduce(lcp_calls_vec.begin(),
                                      lcp_calls_vec.end(), 0, thrust::plus<>());
  uint32_t lcp_sums = thrust::reduce(lcp_sums_vec.begin(), lcp_sums_vec.end(),
                                     0, thrust::plus<>());
  printf("lcp calls: %u, lcp sums: %u\n", lcp_calls, lcp_sums);
  // compute the average lcp
  if (lcp_calls > 0) {
    float avg_lcp = static_cast<float>(lcp_sums) / lcp_calls;
    printf("average lcp: %f\n", avg_lcp);
  } else {
    printf("no lcp calls were made\n");
  }
}

void get_factors_directly_from_factors(
    thrust::device_vector<uint8_t>& R, thrust::device_vector<uint8_t>& S,
    thrust::device_vector<uint>& SA_R, thrust::device_vector<uint>& SA_S,
    thrust::device_vector<int>& LCP_R, thrust::device_vector<int>& LCP_S,
    thrust::device_vector<int>& offsets_and_lens,
    thrust::device_vector<int>& MS, int chunck_size) {
  uint* SA_R_ptr = thrust::raw_pointer_cast(SA_R.data());
  uint* SA_S_ptr = thrust::raw_pointer_cast(SA_S.data());
  int* LCP_R_ptr = thrust::raw_pointer_cast(LCP_R.data());
  int* LCP_S_ptr = thrust::raw_pointer_cast(LCP_S.data());
  uint8_t* R_ptr = thrust::raw_pointer_cast(R.data());
  uint8_t* S_ptr = thrust::raw_pointer_cast(S.data());
  int* offsets_ptr = thrust::raw_pointer_cast(offsets_and_lens.data());

  int* MS_ptr = thrust::raw_pointer_cast(MS.data());
  int n_r = R.size();
  int n_s = S.size();
  int thread_nums = divup(n_s, chunck_size);
  thrust::device_vector<uint32_t> lcp_calls_vec(thread_nums, 0);
  thrust::device_vector<uint32_t> lcp_sums_vec(thread_nums, 0);
  uint32_t* lcp_calls_ptr = thrust::raw_pointer_cast(lcp_calls_vec.data());
  uint32_t* lcp_sums_ptr = thrust::raw_pointer_cast(lcp_sums_vec.data());
  auto r = thrust::counting_iterator<int>(0);
  /* printf("thread_nums: %d, n_s: %d, chunck_size: %d, n_r: %d\n", thread_nums,
  n_s, chunck_size, n_r);
  */
  thrust::for_each(r, r + thread_nums, [=] __device__(int i) {
    if (i * chunck_size >= n_s) return;
    int s_offset = i * chunck_size;
    int matching_factor = offsets_ptr[i * 2];
    int lcp_mf = offsets_ptr[i * 2 + 1];
    int prev_lcp = 0;
    bool first = false;
    int i_s = s_offset;
    int i_r = matching_factor;
    size_t suff_S;
    size_t suff_R;
    int prev_i_r = -1;
    int l = lcp_mf;
    // l = l + lcp(suff_R + l, suff_S + l, R_ptr, S_ptr, n_r, n_s);
    // prev_lcp = l;
    while (i_s < min(n_s, s_offset + chunck_size) /*  && i_r < n_r */) {
      suff_S = SA_S_ptr[i_s];
      suff_R = SA_R_ptr[i_r];
      while (S_ptr[suff_S + l] > R_ptr[suff_R + l] && i_r < n_r) {
        first = true;
        prev_i_r = i_r;
        prev_lcp = l;
        i_r++;
        if (i_r >= n_r) break;
        suff_R = SA_R_ptr[i_r];
        int lcp_R = LCP_R_ptr[i_r];
        while (lcp_R > l) {
          prev_i_r = i_r;
          i_r++;
          if (i_r >= n_r) break;
          lcp_R = LCP_R_ptr[i_r];
        }
        if (i_r >= n_r) break;
        suff_R = SA_R_ptr[i_r];
        if (lcp_R == l) {
          // this might be the right bracket, if so, current l will be lcp of
          // the left bracket but the lcp with the right bracket will be at
          // least the lcp of left bracket, we could just drop it here
          l = l + lcp(suff_R + l, suff_S + l, R_ptr, S_ptr, n_r, n_s);
          lcp_calls_ptr[i]++;
          lcp_sums_ptr[i] += l - lcp_R;
          break;
        } else if (lcp_R < l) {
          // suffix pointed to by i_r is lexicographically larger then s_i
          // suffix and is the right bracket, but the left bracket has a
          // larger lcp with the s_i suffix and thus is the matching factor.
          prev_lcp = l;
          l = lcp_R;
          break;
        }
      }
      while (S_ptr[suff_S + l] < R_ptr[suff_R + l] &&
             i_s < min(n_s, s_offset + chunck_size)) {
        first = false;
        if (prev_lcp >= l && prev_i_r >= 0) {
          MS_ptr[suff_S * 2] = SA_R_ptr[prev_i_r];  // pos of the left bracket
          MS_ptr[suff_S * 2 + 1] = prev_lcp;        // len of the MS
        } else {
          MS_ptr[suff_S * 2] = suff_R;  // pos of the MS (right bracket)
          MS_ptr[suff_S * 2 + 1] = l;   // len of the MS
        }
        ++i_s;
        if (i_s >= min(n_s, s_offset + chunck_size)) break;
        suff_S = SA_S_ptr[i_s];
        int lcp_S = LCP_S_ptr[i_s];
        prev_lcp = min(lcp_S, prev_lcp);
        while (lcp_S > l) {
          if (prev_lcp >= l && prev_i_r >= 0) {
            MS_ptr[suff_S * 2] = SA_R_ptr[prev_i_r];  // pos of the MS
            MS_ptr[suff_S * 2 + 1] = prev_lcp;        // len of the MS
          } else {
            MS_ptr[suff_S * 2] = suff_R;  // pos of the MS (right bracket)
            MS_ptr[suff_S * 2 + 1] = l;   // len of the MS
          }
          ++i_s;
          if (i_s >= min(n_s, s_offset + chunck_size)) break;
          lcp_S = LCP_S_ptr[i_s];
          suff_S = SA_S_ptr[i_s];
          prev_lcp = min(lcp_S, prev_lcp);
        }
        if (i_s >= min(n_s, s_offset + chunck_size)) break;

        if (lcp_S == l) {
          suff_S = SA_S_ptr[i_s];
          l = l + lcp(suff_R + l, suff_S + l, R_ptr, S_ptr, n_r, n_s);
          lcp_calls_ptr[i]++;
          lcp_sums_ptr[i] += l - lcp_S;
          break;
        } else if (lcp_S < l) {
          // suff_S belongs to another interval in SA_R
          l = lcp_S;
          prev_lcp = l;
          break;
        }
      }
      while (i_r >= n_r && i_s < min(n_s, s_offset + chunck_size)) {
        suff_S = SA_S_ptr[i_s];
        MS_ptr[suff_S * 2] = SA_R_ptr[prev_i_r];  // pos of the MS
        MS_ptr[suff_S * 2 + 1] =
            first ? l : min(l, LCP_S_ptr[i_s]);  // len of the MS
        if (first) first = false;
        ++i_s;
        if (i_s >= min(n_s, s_offset + chunck_size)) break;
        l = min(l, LCP_S_ptr[i_s]);
      }
    }
  });
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  // sum the lcp calls and ans sums
  uint32_t lcp_calls = thrust::reduce(lcp_calls_vec.begin(),
                                      lcp_calls_vec.end(), 0, thrust::plus<>());
  uint32_t lcp_sums = thrust::reduce(lcp_sums_vec.begin(), lcp_sums_vec.end(),
                                     0, thrust::plus<>());
  printf("lcp calls: %u, lcp sums: %u\n", lcp_calls, lcp_sums);
  // compute the average lcp
  if (lcp_calls > 0) {
    float avg_lcp = static_cast<float>(lcp_sums) / lcp_calls;
    printf("average lcp: %f\n", avg_lcp);
  } else {
    printf("no lcp calls were made\n");
  }
}