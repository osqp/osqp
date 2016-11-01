# Configuration of make process in osqp.mk
include osqp.mk

# Add includes
CFLAGS += -Iinclude

# target executable
TARGETS = $(OUT)/osqp_demo

# Tests
TEST_TARGETS = $(OUT)/osqp_tester  # Add tests for linear algebra functions
TEST_INCLUDES = -Itests/c

# Define objects to compile
OSQP_OBJECTS = src/osqp.o src/cs.o src/util.o src/lin_alg.o src/kkt.o

# Define source and include files
SRC_FILES = $(wildcard src/*.c)
INC_FILES = $(wildcard include/*.h)

# SuiteSparse
SUITESPARSE_DIR = linsys/direct/external/suitesparse
CFLAGS += -I$(SUITESPARSE_DIR) -I$(SUITESPARSE_DIR)/amd/include -I$(SUITESPARSE_DIR)/ldl/include
AMD_SRC_FILES = $(wildcard $(SUITESPARSE_DIR)/amd/src/amd_*.c)
AMD_OBJECTS = $(AMD_SRC_FILES:.c=.o)
SUITESPARSE_OBJS = $(SUITESPARSE_DIR)/ldl/src/ldl.o $(AMD_OBJECTS)

# Compile all C code
.PHONY: default
default: $(TARGETS) $(OUT)/libosqp.a
	@echo "********************************************************************"
	@echo "Successfully compiled OSQP!"
	@echo "Copyright ...."
	@echo "To try the demo, type '$(OUT)/osqp_demo'"
	@echo "********************************************************************"


# For every object file file compile relative .c file in src/
# -c flag tells the compiler to stop after the compilation phase without linking
%.o: src/%.c
	$(CC) $(CFLAGS) -c $< -o $@

# Define OSQP objects dependencies
src/osqp.o: $(SRC_FILES) $(INC_FILES)
src/util.o	: src/util.c include/util.h
src/lin_alg.o: src/lin_alg.c  include/lin_alg.h
src/kkt.o: src/kkt.c  include/kkt.h
src/cs.o: src/cs.c include/cs.h

# Build osqp library
$(OUT)/libosqp.a: $(OSQP_OBJECTS) $(SUITESPARSE_OBJS)
	mkdir -p $(OUT)   # Create output directory
	$(ARCHIVE) $@ $^  # Create archive of objects
	- $(RANLIB) $@    # Add object files in static library and create index

# Build osqp target (demo file)
$(OUT)/osqp_demo: examples/c/osqp_demo.c $(OUT)/libosqp.a
	$(CC) $(CFLAGS) $^ -o $@  $(LDFLAGS)

# Build tests
.PHONY: test
test: $(TEST_TARGETS)
	@echo "********************************************************************"
	@echo "Successfully compiled tests!"
	@echo "To try the tests, type '$(OUT)/osqp_tester'"
	@echo "********************************************************************"

$(OUT)/osqp_tester: tests/c/osqp_tester.c $(OUT)/libosqp.a
	cd tests/c/lin_alg/; julia generate_mat.jl
	$(CC) $(CFLAGS) $(TEST_INCLUDES) $^ -o $@  $(LDFLAGS)


.PHONY: clean
clean:
	@rm -rf $(TARGETS) $(OSQP_OBJECTS) $(SUITESPARSE_OBJS)
	@rm -rf $(OUT)/*.dSYM
purge: clean
	@rm -rf $(OUT)
