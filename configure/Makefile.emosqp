# Simple makefile to compile the OSQP embedded code into a static library


# libmath is needed for the mode 2 embedded code (sqrt specifically).
# This can be removed for type 1 embedded mode.
LDLIBS = -lm

# The OSQP code needs the private and public include directories.
# The code also needs the osqp_configure.h header that is generated in the code,
# so it should be copied into this directory before building.
INCLUDES = -I./inc/public -I./inc/private -I./

.PHONY: all
all: emosqp

srcs := $(wildcard src/*.c)
objs := $(patsubst %.c,%.o,$(srcs))

pubhead := $(wildcard inc/public/*.h)
privhead := $(wildcard inc/private/*.h)

%.o: %.c $(pubhead) $(privhead) osqp_configure.h
	gcc -c $< -o $@ $(INCLUDES)

libemosqp.a: $(objs)
	ar rcs $@ $^

wrksrc := $(wildcard *workspace.c)
wrkhead := $(wildcard *workspace.h)

emosqp: libemosqp.a emosqp.c $(wrksrc) $(wrkhead)
	gcc -o emosqp emosqp.c $(wrksrc) libemosqp.a $(INCLUDES) $(LDLIBS)

.PHONY: clean
clean:
	-rm -f *.o
	-rm -f src/*.o
	-rm -f emosqp
	-rm -f libemosqp.a
