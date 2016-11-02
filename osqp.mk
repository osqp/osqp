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


# Linear systems directories
LINSYS = lin_sys
DIRSRC = $(LINSYS)/direct
DIRSRCEXT = $(DIRSRC)/external


# Optional FLAGS to be passed to the C code
#-------------------------------------------------------------------------------
# Set Print Levels
# 0: no prints
# 1: only final info
# 2: progress print per iteration
# 3: debug level, enables print & dump fcns
PRINTLEVEL = 3
OPT_FLAGS = -DPRINTLEVEL=$(PRINTLEVEL)

# Set Profiling
# 0: no timing information
# 1: runtime (divided in setup and solve)
# 2: detailed profiling
PROFILING = 1
OPT_FLAGS += -DPROFILING=$(PROFILING)


# Add Optional Flags to CFLAGS
CFLAGS += $(OPT_FLAGS)
