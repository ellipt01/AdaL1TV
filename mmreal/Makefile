CC      = icx
AR      = ar
RM      = rm -f
INSTALL = install

DESTLIBDIR = ../lib

OPENMP_FLG = -qopenmp
OPTFLAGS   = -O3 -march=native

CFLAGS   = $(OPTFLAGS) $(OPENMP_FLG) -MMD -MP
CPPFLAGS = -I./include

LIBSRC_OBJS = \
src/mmio.o \
src/mmreal.o

TARGET_LIB = libmmreal.a

.PHONY: all clean install uninstall

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

-include $(LIBSRC_OBJS:.o=.d)
