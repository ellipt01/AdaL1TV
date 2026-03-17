CC      = icx
AR      = ar
RM      = rm -f
INSTALL = install

DESTLIBDIR = ../lib

OPENMP_FLG = -qopenmp
OPTFLAGS   = -O3 -march=native -fp-model=fast

CFLAGS   = $(OPTFLAGS) $(OPENMP_FLG) -MMD -MP
CPPFLAGS = -I./include

LIBSRC_OBJS = \
src/calc.o \
src/grid.o \
src/kernel.o \
src/vector3d.o \
src/data_array.o \
src/io.o \
src/mgcal.o \
src/source.o \
src/private/util.o

TARGET_LIB = libmgcal.a

.PHONY: all install uninstall clean

all: $(TARGET_LIB)

$(TARGET_LIB): $(LIBSRC_OBJS)
	$(AR) rcs $@ $^

install: all
	mkdir -p $(DESTLIBDIR)
	$(INSTALL) -m 644 $(TARGET_LIB) $(DESTLIBDIR)

uninstall:
	$(RM) $(DESTLIBDIR)/$(TARGET_LIB)

clean:
	$(RM) $(LIBSRC_OBJS) $(LIBSRC_OBJS:.o=.d) $(TARGET_LIB)

%.o: %.c
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

-include $(LIBSRC_OBJS:.o=.d)
