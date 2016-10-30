#ifndef CONSTANTS_H
#define CONSTANTS_H






/* DEFAULT SOLVER PARAMETERS AND SETTINGS --------------------------------    */
/* TODO: complete parameters, these are just a couple of examples */
#define PRINTLEVEL (3)     /* 0: no prints					             */
						   /* 1: only final info				         */
                           /* 2: progress print per iteration            */
						   /* 3: debug level, enables print & dump fcns. */
#define MAX_ITERS (2500)



/* DATA CUSTOMIZATIONS ----------------------------------------------------   */
/* define custom printfs and memory allocation as needed (e.g. matlab or python) */
#include <stdlib.h>
#define c_malloc malloc
#define c_calloc calloc
#define c_free free

/* Use customized number representation ---------- */
typedef int c_int;                   /* for indeces */
typedef double c_float;              /* for numerical values  */

/* Use customized operations */
#define c_sqrt sqrt  // Doubles
#define c_sqrtf sqrtf  // Floats


/* Custom printing functins ------------------------------------------------- */
#if PRINTLEVEL > 0
#define c_print printf
#endif







#endif
