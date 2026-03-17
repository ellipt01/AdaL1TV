# =========================================
# MKL
# =========================================
MKLROOT ?= /opt/intel/oneapi/mkl/latest

BLAS_LIB    = `pkg-config mkl-dynamic-lp64-iomp --define-variable=MKLROOT=${MKLROOT} --libs`
BLAS_CFLAGS = `pkg-config mkl-dynamic-lp64-iomp --define-variable=MKLROOT=${MKLROOT} --cflags`

OPENMP_FLG  = -qopenmp

# =========================================
# Compiler
# =========================================
CC  = icx
CXX = icpx
MAKE = make
RM   = rm -f
INSTALL = install

# =========================================
# Directories
# =========================================
LIBDIR  = ./lib
DESTDIR = ./bin

# =========================================
# Flags
# =========================================
OPTFLAGS = -O3 -march=native -fp-model=fast $(OPENMP_FLG)

CFLAGS   = $(OPTFLAGS) -MMD -MP
CXXFLAGS = $(OPTFLAGS) -MMD -MP

CPPFLAGS = \
-I./include \
-I./mgcal/include \
-I./mmreal/include \
$(BLAS_CFLAGS)

ifeq ($(TIMING),1)
	CPPFLAGS += -DENABLE_TIMING
endif

LIBS = \
-L$(LIBDIR) \
-lmgcal \
-lmmreal \
$(BLAS_LIB) \
-lm

# =========================================
# Objects
# =========================================
COMMON_OBJS = \
src/L1L2.o \
src/ADMM.o \
src/MagKernel.o \
src/Kernel.o \
src/DiffOp.o \
src/PardisoSolver.o

L1L2_OBJS = src/l1l2inv.o

ADAL1TV_OBJS = \
src/l1tvinv.o \
src/AdaL1TV.o \
src/ADMM_AdaL1TV.o

PROGRAMS = l1l2inv l1tvinv

SUBDIRS = mgcal mmreal

# =========================================
# Targets
# =========================================
.PHONY: all clean install uninstall $(SUBDIRS)

all: $(SUBDIRS) $(PROGRAMS)

# --- build sub libraries ---
$(SUBDIRS):
	$(MAKE) -C $@
	$(MAKE) install -C $@

# --- executables ---
l1l2inv: $(L1L2_OBJS) $(COMMON_OBJS) | $(SUBDIRS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

l1tvinv: $(ADAL1TV_OBJS) $(COMMON_OBJS) | $(SUBDIRS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

# =========================================
# Compile
# =========================================
%.o: %.c
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

# =========================================
# Install
# =========================================
install: all
	mkdir -p $(DESTDIR)
	for p in $(PROGRAMS); do \
		$(INSTALL) -m 755 $$p $(DESTDIR); \
	done

# =========================================
# Uninstall
# =========================================
uninstall:
	@echo "Removing installed programs..."
	for p in $(PROGRAMS); do \
		$(RM) $(DESTDIR)/$$p ; \
	done
	for d in $(SUBDIRS); do \
		$(MAKE) uninstall -C $$d ; \
	done

# =========================================
# Clean
# =========================================
clean:
	$(RM) src/*.o src/*.d $(PROGRAMS)
	for d in $(SUBDIRS); do \
		$(MAKE) clean -C $$d ; \
	done

-include src/*.d
