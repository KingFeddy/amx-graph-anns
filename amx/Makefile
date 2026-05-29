MKL_ROOT = /opt/intel/oneapi/mkl/2026.0
OMP_ROOT = /opt/intel/oneapi/compiler/2026.0

CXX      = g++
CXXFLAGS = -O3 -std=c++17 -march=native \
           -I$(MKL_ROOT)/include

LDFLAGS  = -L$(MKL_ROOT)/lib \
           -L$(OMP_ROOT)/lib \
           -lmkl_intel_lp64 \
           -lmkl_intel_thread \
           -lmkl_core \
           -liomp5 \
           -lpthread \
           -lm \
           -Wl,-rpath,$(MKL_ROOT)/lib \
           -Wl,-rpath,$(OMP_ROOT)/lib

all: amx_gemm_test

amx_gemm_test: amx_gemm_test.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f amx_gemm_test

amx_scaling_extended: amx_scaling_extended.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)
