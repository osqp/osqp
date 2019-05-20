#ifndef OSQP_CUSTOM_MEMORY_H

#define OSQP_CUSTOM_MEMORY_H

# ifdef __cplusplus
extern "C" {
# endif /* ifdef __cplusplus */

#  include <stdio.h> //for size_t

#  define c_malloc  my_malloc
#  define c_calloc  my_calloc
#  define c_free    my_free
#  define c_realloc my_realloc

/* functions should have the same
signatures as the standard ones */
void* my_malloc(size_t size);
void* my_calloc(size_t num, size_t size);
void* my_realloc(void *ptr, size_t size);
void  my_free(void *ptr);

# ifdef __cplusplus
}
# endif /* ifdef __cplusplus */

#endif /* ifndef  OSQP_CUSTOM_MEMORY_H */
