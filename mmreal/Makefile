# Compiler and tools (override with e.g. make CC=gcc)
CC      = icx
AR      ?= ar
RM      ?= rm -f
INSTALL ?= install

DESTLIBDIR ?= ../lib

# OpenMP flag depends on the compiler family.
ifeq ($(findstring icx,$(CC))$(findstring icc,$(CC)),)
  OPENMP_FLG ?= -fopenmp      # gcc / clang / cc
else
  OPENMP_FLG ?= -qopenmp      # Intel icx / icc
endif

OPTFLAGS   ?= -O3
# -march=native is build-host specific; enable explicitly if desired:
#   make OPTFLAGS="-O3 -march=native"

CFLAGS   += $(OPTFLAGS) $(OPENMP_FLG) -MMD -MP
CPPFLAGS += -I./include

# BLAS integer width.
#   make            -> 32-bit BLAS integers (blas_int = int), LP64 BLAS
#   make ILP64=1    -> 64-bit BLAS integers (blas_int = int64_t), ILP64 BLAS
# NOTE: this Makefile only archives objects (ar); it does not link a BLAS.
# The executable that links libmmreal.a must use the SAME ILP64 setting and
# link the matching MKL interface layer (lp64 vs ilp64).
ILP64 ?= 0
ifeq ($(ILP64),1)
	CPPFLAGS += -DMM_BLAS_ILP64 -DMKL_ILP64
endif

LIBSRC_OBJS = \
	src/mmio.o \
	src/mmreal.o

TARGET_LIB = libmmreal.a

SUBDIRS =

.PHONY: all clean install uninstall $(SUBDIRS)

all: $(TARGET_LIB) $(SUBDIRS)

# --- build sub libraries ---
$(SUBDIRS):
	$(MAKE) -C $@

# Explicit rule (documents how objects are built; matches the implicit one).
src/%.o: src/%.c
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

$(TARGET_LIB): $(LIBSRC_OBJS)
	$(AR) rcs $@ $^

install: all
	mkdir -p $(DESTLIBDIR)
	$(INSTALL) -m 644 $(TARGET_LIB) $(DESTLIBDIR)

uninstall:
	$(RM) $(DESTLIBDIR)/$(TARGET_LIB)

clean:
	$(RM) $(LIBSRC_OBJS) $(LIBSRC_OBJS:.o=.d) $(TARGET_LIB)

-include $(LIBSRC_OBJS:.o=.d)
