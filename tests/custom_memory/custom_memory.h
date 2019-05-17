#ifndef OSQP_CUSTOM_MEMORY_H
#define OSQP_CUSTOM_MEMORY_H

# ifdef __cplusplus
extern "C" {
# endif /* ifdef __cplusplus */


#  include <stdlib.h>
#  include <stdio.h>

#  define c_malloc  my_malloc
#  define c_calloc  my_calloc
#  define c_free    my_free
#  define c_realloc my_realloc

//Make a global counter and track the net number of allocations
//by user defined allocators.   Should go to zero on exit if no leaks.
static long int alloc_counter = 0;

static void* my_malloc(size_t size) {
  void *m = malloc(size);
  alloc_counter++;
  printf("OSQP allocator  (malloc): %zu bytes, %ld allocations \n",size, alloc_counter);
  return m;
}

static void* my_calloc(size_t num, size_t size) {
  void *m = calloc(num, size);
  alloc_counter++;
  printf("OSQP allocator  (calloc): %zu bytes, %ld allocations \n",num*size, alloc_counter);
  return m;
}

static void* my_realloc(void *ptr, size_t size) {
  void *m = realloc(ptr,size);
  printf("OSQP allocator (realloc) : %zu bytes, %ld allocations \n",size, alloc_counter);
  return m;
}

static void my_free(void *ptr) {
  free(ptr);
  alloc_counter--;
  printf("OSQP allocator   (free) : %ld allocations \n", alloc_counter);
}

# ifdef __cplusplus
}
# endif /* ifdef __cplusplus */

#endif /* ifndef  OSQP_CUSTOM_MEMORY_H */
