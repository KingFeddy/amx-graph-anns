MKL_ROOT = /opt/intel/oneapi/mkl/2026.0
OMP_ROOT = /opt/intel/oneapi/compiler/2026.0
CXX      = g++
CXXFLAGS = -O3 -std=c++17 -march=native -I$(MKL_ROOT)/include
LDFLAGS  = -L$(MKL_ROOT)/lib -L$(OMP_ROOT)/lib \
           -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core \
           -liomp5 -lpthread -lm \
           -Wl,-rpath,$(MKL_ROOT)/lib -Wl,-rpath,$(OMP_ROOT)/lib

# All benchmarks below call arch_prctl so BF16 GEMM engages AMX tiles
# (without it MKL silently routes BF16 to AVX-512 — see README).
all: amx_init_test amx_r_sweep

amx_init_test: amx_init_test.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

amx_r_sweep: amx_r_sweep.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

amx_scaling_extended: amx_scaling_extended.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

barrier_overhead_test: barrier_overhead_test.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f amx_init_test amx_r_sweep amx_scaling_extended barrier_overhead_test
