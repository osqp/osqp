# Makefile configuration for OSQP depending on the specific architecture

# Check which operative systems we are running
#----------------------------------------------
ifeq ($(OS),Windows_NT)
UNAME = CYGWINorMINGWorMSYS
else
UNAME = $(shell uname -s)
endif

# Windows
ifeq (CYGWIN, $(findstring CYGWIN, $(UNAME)))
ISWINDOWS := 1
else ifeq (MINGW, $(findstring MINGW, $(UNAME)))
ISWINDOWS := 1
else ifeq (MSYS, $(findstring MSYS, $(UNAME)))
ISWINDOWS := 1
else
ISWINDOWS := 0
endif

# Mac
ifeq ($(UNAME), Darwin)
ISMAC :=1
else
ISMAC :=0
endif

# Unix
ifeq ($(ISMAC), 0)
ifeq ($ISWINDOWS), 0)
ISLINUX := 1
endif
endif


# Setup compiler and flagsompiler
#-----------------

CC = gcc

# compiler flags:
  #  -g    adds debugging information to the executable file
  #  -Wall turns on most, but not all, compiler warnings
  #  -O3 turns on all optimizations specified by -O2 and also turns on the -finline-functions, -fweb and -frename-registers options.
CFLAGS = -g -O3
# CFLAGS += -Wall

# Add coverage
CFLAGS += --coverage

# Enforce ANSI C
# CFLAGS += -ansi

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

# Tests folders
TESTSDIR = tests




# Change flags and parameters according to the operative system
#----------------------------------------------------

# Windows
ifeq ($(ISWINDOWS), 1)
# shared library has extension .dll
SHAREDEXT = dll
SONAME = -soname

else ifeq ($(ISMAC), 1)
# shared library has extension .dylib
SHAREDEXT = dylib
SONAME = -install_name
CFLAGS += -fPIC

else ifeq ($(ISLINUX), 1)
# use accurate timer from clock_gettime()
LDFLAGS += -lrt
# shared library has extension .so
SHAREDEXT = so
SONAME = -soname
CFLAGS += -fPIC

endif



# Optional FLAGS to be passed to the C code
#-------------------------------------------------------------------------------
# Enable printing
OPT_FLAGS = -DPRINTING

# Set Profiling
OPT_FLAGS += -DPROFILING

# Use floats instead of doubles
OPT_FLAGS += -DDFLOAT

# Use long integers for indexing
# OPT_FLAGS += -DDLONG

# Add Optional Flags to CFLAGS
CFLAGS += $(OPT_FLAGS)
