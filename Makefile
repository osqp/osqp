# Configuration of make process in osqp.mk
include osqp.mk

# Add includes
CFLAGS += -Iinclude

# target executable
TARGET = $(OUT)/osqp_demo

# Define objects to compile
OSQP_OBJECTS = src/osqp.o src/lin_alg.o

# Define source and include files
SRC_FILES = $(wildcard src/*.c)
INC_FILES = $(wildcard include/*.h)

# define the C object files
# This uses Suffix Replacement within a macro:
#   $(name:string1=string2)
#         For each word in 'name' replace 'string1' with 'string2'
# Below we are replacing the suffix .c of all words in the macro SRCS
# with the .o suffix
# OBJ_FILES = $(SRCS:.c=.o)


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

# Define objects dependencies
src/osqp.o: $(SRC_FILES) $(INC_FILES)
src/lin_alg.o: src/lin_alg.c  include/lin_alg.h

# Build osqp library
$(OUT)/libosqp.a: $(OSQP_OBJECTS)
	mkdir -p $(OUT)   # Create output directory
	$(ARCHIVE) $@ $^  # Create archive of objects
	- $(RANLIB) $@    # Add object files in static library and create index

# Build target (demo file)
$(OUT)/osqp_demo: examples/c/osqp_demo.c $(OUT)/libosqp.a
	$(CC) $(CFLAGS) $^ -o $@  $(LDFLAGS)



.PHONY: clean
clean:
	@rm -rf $(TARGETS) $(OSQP_OBJECTS)
	@rm -rf $(OUT)/*.dSYM
purge: clean
	@rm -rf $(OUT)





#
#
# .PHONY: default all clean
#
# default: $(TARGET)
# all: default
#
# OBJECTS = $(patsubst %.c, %.o, $(wildcard src/*.c))
# # SRC_FILES = $(wildcard src/*.c)
# # INC_FILES = $(wildcard include/*.h)
#
#
# %.o: src/%.c
# 	$(CC) $(CFLAGS) -c $< -o $@
#
# # Specify dependencies
# hellomake.o: src/hellomake.c include/hellomake.h
#
#
#
# .PRECIOUS: $(TARGET) $(OBJECTS)
#
# $(TARGET): $(OBJECTS)
# 	$(CC) $(OBJECTS) -Wall $(LIBS) -o $@
#
# clean:
# 	-rm -f *.o
# 	-rm -f $(TARGET)
