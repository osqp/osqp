# Makefile configuration for OSQP depending on the specific architecture

# compiler
CC = gcc

# compiler flags:
  #  -g    adds debugging information to the executable file
  #  -Wall turns on most, but not all, compiler warnings
CFLAGS = -g -Wall


# additional library paths
# -lm: Basic math library
LDFLAGS = -lm


# Output directory
OUT = out


# Archive files for linker
ARFLAGS = rv
ARCHIVE = $(AR) $(ARFLAGS)
RANLIB = ranlib
