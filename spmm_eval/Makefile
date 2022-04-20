CC = nvcc

FLAGS = -arch=sm_60 -O3 -std=c++11 -w -maxrregcount=64 -rdc=true
LINK = -lcublas -lcusparse -lcudart

# baseline
baseline: backend/spmm.cu backend/utility.cu backend/csr2b2sr_batch.cu
		$(CC) $(FLAGS) $(LINK) test_spmm_baseline.cu -o spmm/$@

# spmm bin-bin
spmmbb: spmmbb2 spmmbb4 spmmbb8 spmmbb16
spmmbb2: backend/spmm.cu backend/utility.cu backend/csr2b2sr_batch.cu
		$(CC) $(FLAGS) $(LINK) test_spmm_bin_bin.cu -DTILEDIM=4 -DOUTUNIT=2 -o spmm/$@
spmmbb4: backend/spmm.cu backend/utility.cu backend/csr2b2sr_batch.cu
		$(CC) $(FLAGS) $(LINK) test_spmm_bin_bin.cu -DTILEDIM=4 -DOUTUNIT=4 -o spmm/$@
spmmbb8: backend/spmm.cu backend/utility.cu backend/csr2b2sr_batch.cu
		$(CC) $(FLAGS) $(LINK) test_spmm_bin_bin.cu -DTILEDIM=4 -DOUTUNIT=8 -o spmm/$@
spmmbb16: backend/spmm.cu backend/utility.cu backend/csr2b2sr_batch.cu
		$(CC) $(FLAGS) $(LINK) test_spmm_bin_bin.cu -DTILEDIM=4 -DOUTUNIT=16 -o spmm/$@

# spmm full-bin
spmmfb: spmmfb2 spmmfb4 spmmfb8 spmmfb16
spmmfb2: backend/spmm.cu backend/utility.cu backend/csr2b2sr_batch.cu
		$(CC) $(FLAGS) $(LINK) test_spmm_full_bin.cu -DTILEDIM=4 -DOUTUNIT=2 -o spmm/$@
spmmfb4: backend/spmm.cu backend/utility.cu backend/csr2b2sr_batch.cu
		$(CC) $(FLAGS) $(LINK) test_spmm_full_bin.cu -DTILEDIM=4 -DOUTUNIT=4 -o spmm/$@
spmmfb8: backend/spmm.cu backend/utility.cu backend/csr2b2sr_batch.cu
		$(CC) $(FLAGS) $(LINK) test_spmm_full_bin.cu -DTILEDIM=4 -DOUTUNIT=8 -o spmm/$@
spmmfb16: backend/spmm.cu backend/utility.cu backend/csr2b2sr_batch.cu
		$(CC) $(FLAGS) $(LINK) test_spmm_full_bin.cu -DTILEDIM=4 -DOUTUNIT=16 -o spmm/$@

# spmm full-full
spmmff: spmmff1 spmmff2 spmmff4 spmmff8
spmmff1: backend/spmm.cu backend/utility.cu backend/csr2b2sr_batch.cu
		$(CC) $(FLAGS) $(LINK) test_spmm_full_full.cu -DTILEDIM=4 -DOUTUNIT=1 -o spmm/$@
spmmff2: backend/spmm.cu backend/utility.cu backend/csr2b2sr_batch.cu
		$(CC) $(FLAGS) $(LINK) test_spmm_full_full.cu -DTILEDIM=4 -DOUTUNIT=2 -o spmm/$@
spmmff4: backend/spmm.cu backend/utility.cu backend/csr2b2sr_batch.cu
		$(CC) $(FLAGS) $(LINK) test_spmm_full_full.cu -DTILEDIM=4 -DOUTUNIT=4 -o spmm/$@
spmmff8: backend/spmm.cu backend/utility.cu backend/csr2b2sr_batch.cu
		$(CC) $(FLAGS) $(LINK) test_spmm_full_full.cu -DTILEDIM=4 -DOUTUNIT=8 -o spmm/$@

clean: 
	rm -f spmm/*