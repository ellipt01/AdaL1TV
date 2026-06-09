# ======================================================================
# Standalone Makefile for the mmreal test suite.
#
#   make            build and run the tests
#   make build      build the test binary only
#   make run        run the test binary
#   make clean      remove build artifacts
#
# Configuration (override on the command line):
#
#   make CC=gcc                          choose the compiler
#   make SRC_DIR=../src INC_DIR=../include
#   make BLAS_LIB="-lopenblas"           link a specific BLAS
#   make ILP64=1                         64-bit BLAS integers (needs ILP64 BLAS)
#
# By default this builds the library sources together with the tests using a
# reference BLAS (-lblas) and 32-bit BLAS integers, which is the portable,
# vendor-neutral configuration.
# ======================================================================

CC       = icx

# Where the library sources and headers live, relative to this directory.
SRC_DIR  ?= ../src
INC_DIR  ?= ../include

# BLAS to link against. Override with e.g. BLAS_LIB="-lopenblas".
BLAS_LIB ?= `pkg-config mkl-dynamic-lp64-iomp --define-variable=MKLROOT=${MKLROOT} --libs`

# C standard: C11 is required (thread_local, stdbool).
STD      ?= -std=c11

# Warning flags. -Werror is on by default so that any blas_int pointer
# mismatch or other regression fails the build loudly.
WARN     ?= -Wall -Wextra -Werror

OPTFLAGS ?= -O2 -g

# OpenMP: mmreal.c uses #pragma omp in its matrix-product routines. Without
# this flag the compiler warns about unknown pragmas (and -Werror fails).
# OpenMP flag depends on the compiler family.
ifeq ($(findstring icx,$(CC))$(findstring icc,$(CC)),)
  OPENMP_FLG ?= -fopenmp      # gcc / clang / cc
else
  OPENMP_FLG ?= -qopenmp      # Intel icx / icc
endif

CPPFLAGS += -I$(INC_DIR) -I.
CFLAGS   += $(STD) $(WARN) $(OPTFLAGS) $(OPENMP)
LDFLAGS  += $(OPENMP)

# 64-bit BLAS integers (only when linking an ILP64 BLAS).
ILP64 ?= 0
ifeq ($(ILP64),1)
  CPPFLAGS += -DMM_BLAS_ILP64
endif

# Math library and BLAS.
LDLIBS += $(BLAS_LIB) -lm

# Library sources compiled directly into the test binary (no install needed).
LIB_SRCS = $(SRC_DIR)/mmio.c $(SRC_DIR)/mmreal.c
TEST_SRC = test_mmreal.c
OBJS     = $(LIB_SRCS:.c=.o) $(TEST_SRC:.c=.o)

TEST_BIN = test_mmreal

.PHONY: all build run clean
all: run

build: $(TEST_BIN)

$(TEST_BIN): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $@ $(LDLIBS)

# Generic compile rule (works for both ../src/*.c and ./test_mmreal.c).
%.o: %.c
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

run: $(TEST_BIN)
	./$(TEST_BIN)

clean:
	$(RM) $(OBJS) $(TEST_BIN)

install:

