# Configuration of make process in osqp.mk
include osqp.mk

# Add includes
CFLAGS += -Iinclude

# target executable
TARGET = $(OUT)/osqp_demo 
TARGET += $(OUT)/test_lin_alg  # Add tests for linear algebra functions

# Define objects to compile
OSQP_OBJECTS = src/osqp.o src/lin_alg.o src/cs.o

# Define source and include files
SRC_FILES = $(wildcard src/*.c)
INC_FILES = $(wildcard include/*.h)


# Compile all C code
.PHONY: default
default: $(TARGET) $(OUT)/libosqp.a
	@echo "****************************************************************************************"
	@echo "Successfully compiled OSQP!"
	@echo "Copyright ...."
	@echo "To test, type '$(OUT)/osqp_demo'"
	@echo "****************************************************************************************"


# For every object file file compile relative .c file in src/
%.o : src/%.c
	$(CC) $(CFLAGS) -c $< -o $@

# Define OSQP objects dependencies
src/osqp.o: $(SRC_FILES) $(INC_FILES)
src/lin_alg.o: src/lin_alg.c  include/lin_alg.h
src/cs.o	: src/cs.c include/cs.h

# Build osqp library
$(OUT)/libosqp.a: $(OSQP_OBJECTS)
	mkdir -p $(OUT)   # Create output directory
	$(ARCHIVE) $@ $^  # Create archive of objects
	- $(RANLIB) $@    # Add object files in static library and create index

# Build target (demo file)
$(OUT)/osqp_demo: examples/c/osqp_demo.c $(OUT)/libosqp.a
	$(CC) $(CFLAGS) $^ -o $@  $(LDFLAGS)
	
# Build target (linear algebra tests)
$(OUT)/tests_lin_alg: examples/c/tests_lin_alg.c examples/c/tests_matrices/matrices.h $(OUT)/libosqp.a
	$(CC) $(CFLAGS) $^ -o $@  $(LDFLAGS)



.PHONY: clean
clean:
	@rm -rf $(TARGETS) $(OSQP_OBJECTS)
	@rm -rf $(OUT)/*.dSYM
purge: clean
	@rm -rf $(OUT)

