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
OSQP_OBJECTS = src/util.o src/auxil.o src/cs.o src/lin_alg.o src/kkt.o src/proj.o src/scaling.o src/polish.o src/osqp.o

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
%.o: src/%.c
	$(CC) $(CFLAGS) -c $< -o $@

# Define OSQP objects dependencies
src/osqp.o: $(SRC_FILES) $(INC_FILES)
src/cs.o: src/cs.c include/cs.h
src/lin_alg.o: src/lin_alg.c  include/lin_alg.h
src/lin_sys.o: src/lin_sys.c  include/lin_sys.h
src/kkt.o	: src/kkt.c include/kkt.h
src/util.o	: src/util.c include/util.h
src/auxil.o	: src/auxil.c include/auxil.h
src/polish.o	: src/polish.c include/polish.h
src/scaling.o	: src/scaling.c include/scaling.h



# Define linear systems solvers objects and dependencies
# Direct
$(DIRSRC)/private.o: $(DIRSRC)/private.c  $(DIRSRC)/private.h


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


$(OUT)/osqp_tester_direct: tests/c/osqp_tester_direct.c $(OUT)/libosqpdir.a $(TEST_OBJECTS)
	# cd tests/c/; julia generate_tests.jl
	$(CC) $(CFLAGS) $(TEST_INCLUDES) $^ -o $@  $(LDFLAGS)


# Create coverage statistics
# 1) Compile test code
# 2) run tester
# 3) Analyze results with lcov
# 4) Export results to coverage_html
.PHONY: coverage
coverage: $(OUT)/osqp_tester_direct
	out/osqp_tester_direct
	@lcov --capture --directory . --output-file coverage.info
	@lcov --remove coverage.info 'lin_sys/direct/external/suitesparse/*' 'tests/*' -o coverage.info
	@genhtml coverage.info --output-directory coverage_html



.PHONY: clean
clean:
	@rm -rf $(TARGETS) $(OSQP_OBJECTS) $(SUITESPARSE_OBJS) $(LINSYS)/*.o $(DIRSRC)/*.o
	@rm -rf $(OUT)/*.dSYM
	@rm -rf $(TEST_OBJECTS)
	@rm -rf coverage.info coverage_html
	@find . -name "*.gcno" -type f -delete
	@find . -name "*.gcda" -type f -delete
purge: clean
	@rm -rf $(OUT)
