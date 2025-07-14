NVCC=nvcc

all: ms-by-brackets lcp-testing ms-by-merge ms-by-merge-chunked ms-by-merge-chunked-sa ms-by-brackets-chunked lmp-ms

ms-by-brackets: ms_brackets.cu ms_utils.cu utils.cu lcp_utils.cu libcu_utils.cu 
	$(NVCC) --extended-lambda --expt-relaxed-constexpr -o ms-by-brackets ms_brackets.cu

lcp-testing: lcp.cu utils.cu lcp_utils.cu libcu_utils.cu 
	$(NVCC) --extended-lambda --expt-relaxed-constexpr  -o lcp-testing lcp.cu

ms-by-merge: ms_by_merge.cu ms_utils.cu utils.cu libcu_utils.cu lcp_utils.cu merge_utils.cu
	$(NVCC) --extended-lambda --expt-relaxed-constexpr -o ms-by-merge ms_by_merge.cu

ms-by-merge-chunked: ms_by_merge_chunked.cu ms_utils.cu utils.cu lcp_utils.cu merge_utils.cu libcu_utils.cu
	$(NVCC) --extended-lambda --expt-relaxed-constexpr  -o ms-by-merge-chunked ms_by_merge_chunked_sa.cu

ms-by-merge-chunked-sa: ms_by_merge_chunked_sa.cu utils.cu lcp_utils.cu merge_utils.cu libcu_utils.cu
	$(NVCC) --extended-lambda --expt-relaxed-constexpr -o ms-by-merge-chunked-sa ms_by_merge_chunked_sa.cu

ms-by-brackets-chunked: brackets_chunked.cu utils.cu lcp_utils.cu ms_utils.cu libcu_utils.cu libcubwt.cu
	$(NVCC) --extended-lambda --expt-relaxed-constexpr -o ms-by-brackets-chunked brackets_chunked.cu

lmp-ms: lmp_ms.cu utils.cu lmp_utils.cu lcp_utils.cu libcu_utils.cu
	$(NVCC) --extended-lambda --expt-relaxed-constexpr  -o lmp-ms lmp_ms.cu

clean:
	/bin/rm -f /*.o

nuclear:
	/bin/rm -f ms-by-brackets lcp-testing ms-by-merge ms-by-merge-chunked ms-by-merge-chunked-sa ms-by-brackets-chunked lmp-ms