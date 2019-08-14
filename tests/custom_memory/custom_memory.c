#  include "custom_memory.h"
#  include <stdlib.h>
#  include <stdio.h>

//Make a global counter and track the net number of allocations
//by user defined allocators.   Should go to zero on exit if no leaks.
long int alloc_counter = 0;

void* my_malloc(size_t size) {
  void *m = malloc(size);
  alloc_counter++;
  /* printf("OSQP allocator  (malloc): %zu bytes, %ld allocations \n",size, alloc_counter); */
  return m;
}

void* my_calloc(size_t num, size_t size) {
  void *m = calloc(num, size);
  alloc_counter++;
  /* printf("OSQP allocator  (calloc): %zu bytes, %ld allocations \n",num*size, alloc_counter); */
  return m;
}

void* my_realloc(void *ptr, size_t size) {
  void *m = realloc(ptr,size);
  /* printf("OSQP allocator (realloc) : %zu bytes, %ld allocations \n",size, alloc_counter); */
  return m;
}

void my_free(void *ptr) {
  if(ptr != NULL){
    free(ptr);
    alloc_counter--;
    /* printf("OSQP allocator   (free) : %ld allocations \n", alloc_counter); */
  }
}
