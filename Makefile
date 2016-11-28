# Configuration of make process in osqp.mk
include osqp.mk

# Add includes
CFLAGS += -Iinclude

# target executable
TARGETS = $(OUT)/osqp_demo_direct

# Tests
TEST_TARGETS = $(OUT)/osqp_tester_direct  # Add tests for linear algebra functions
TEST_INCLUDES = -I$(TESTSDIR)/c
# OnlineQP Lib tests
QPTESTSDIR = $(TESTSDIR)/c/qptests
# TEST_OBJECTS = $(QPTESTSDIR)/chain80w/chain80w.o
TEST_OBJECTS = $(QPTESTSDIR)/diesel/diesel.o

# Define objects to compile
OSQP_OBJECTS = src/util.o src/aux.o src/cs.o src/lin_alg.o src/kkt.o src/scaling.o src/polish.o  src/osqp.o

# Define source and include files
SRC_FILES = $(wildcard src/*.c)
INC_FILES = $(wildcard include/*.h)

# SuiteSparse
SUITESPARSE_DIR = $(DIRSRCEXT)/suitesparse
CFLAGS += -I$(SUITESPARSE_DIR) -I$(SUITESPARSE_DIR)/amd/include -I$(SUITESPARSE_DIR)/ldl/include
AMD_SRC_FILES = $(wildcard $(SUITESPARSE_DIR)/amd/src/amd_*.c)
AMD_OBJECTS = $(AMD_SRC_FILES:.c=.o)
SUITESPARSE_OBJS = $(SUITESPARSE_DIR)/SuiteSparse_config.o $(SUITESPARSE_DIR)/ldl/src/ldl.o $(AMD_OBJECTS)

# Compile all C code
.PHONY: default
default: $(TARGETS) $(OUT)/libosqpdir.a
	@echo "********************************************************************"
	@echo "Successfully compiled OSQP!"
	@echo "Copyright ...."
	@echo "To try the demo, type '$(OUT)/osqp_demo_direct'"
	@echo "********************************************************************"


# For every object file file compile relative .c file in src/
# -c flag tells the compiler to stop after the compilation phase without linking
# %.o: src/%.c
#	 $(CC) $(CFLAGS) -c $< -o $@

# Define OSQP objects dependencies
# src/osqp.o: $(SRC_FILES) $(INC_FILES)
# src/util.o	: src/util.c include/util.h
# src/lin_alg.o: src/lin_alg.c  include/lin_alg.h
# src/lin_sys.o: src/lin_sys.c  include/lin_sys.h
# src/cs.o: src/cs.c include/cs.h


# Define linear systems solvers objects and dependencies
# Direct
# $(DIRSRC)/private.o: $(DIRSRC)/private.c  $(DIRSRC)/private.h


# Build osqp library (direct method)
$(OUT)/libosqpdir.a: $(OSQP_OBJECTS) $(DIRSRC)/private.o $(SUITESPARSE_OBJS) $(LINSYS)/common.o
	mkdir -p $(OUT)   # Create output directory
	$(ARCHIVE) $@ $^  # Create archive of objects
	- $(RANLIB) $@    # Add object files in static library and create index

# Build osqp target (demo file for direct method)
$(OUT)/osqp_demo_direct: examples/osqp_demo_direct.c $(OUT)/libosqpdir.a
	$(CC) $(CFLAGS) $^ -o $@  $(LDFLAGS)

# Build tests
.PHONY: test
test: $(TEST_TARGETS)
	@echo "********************************************************************"
	@echo "Successfully compiled tests!"
	@echo "To try the tests, type '$(OUT)/osqp_tester_direct'"
	@echo "********************************************************************"

# $(QPTESTSDIR)/chain80w/chain80w.o: $(QPTESTSDIR)/chain80w/chain80w.c $(QPTESTSDIR)/chain80w/chain80w.h
# 	@echo "Vaffa!"

$(OUT)/osqp_tester_direct: tests/c/osqp_tester_direct.c $(OUT)/libosqpdir.a $(TEST_OBJECTS)
	# cd tests/c/; julia generate_tests.jl
	$(CC) $(CFLAGS) $(TEST_INCLUDES) $^ -o $@  $(LDFLAGS)




.PHONY: clean
clean:
	@rm -rf $(TARGETS) $(OSQP_OBJECTS) $(SUITESPARSE_OBJS) $(LINSYS)/*.o $(DIRSRC)/*.o
	@rm -rf $(OUT)/*.dSYM
	@rm -rf $(TEST_OBJECTS)
purge: clean
	@rm -rf $(OUT)
