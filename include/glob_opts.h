#ifndef GLOB_OPTS_H
#define GLOB_OPTS_H

/* DATA CUSTOMIZATIONS (depending on memory manager)-----------------------   */
/* define custom printfs and memory allocation (e.g. matlab or python) */
#ifdef MATLAB_MEX_FILE
    #include "mex.h"
    static void* c_calloc(size_t num, size_t size){
        void *m = mxCalloc(num,size);
        mexMakeMemoryPersistent(m);
        return m;
    }
    static void* c_malloc(size_t size){
        void *m = mxMalloc(size);
        mexMakeMemoryPersistent(m);
        return m;
    }
    static void* c_realloc(void *ptr, size_t size){
        void *m = mxRealloc(ptr,size);
        mexMakeMemoryPersistent(m);
        return m;
    }
    #define c_free mxFree
#elif defined PYTHON
    // Define memory allocation for python. Note that in Python 2 memory manager
    // Calloc is not implemented
    #include <Python.h>
    #define c_malloc PyMem_Malloc
    #define c_calloc(n,s) ({                     \
            void * p_calloc = c_malloc((n)*(s)); \
            memset(p_calloc, 0, (n)*(s));        \
            p_calloc;                            \
        })
    #define c_free PyMem_Free
    #define c_realloc PyMem_Realloc
#else
    #define c_malloc malloc
    #define c_calloc calloc
    #define c_free free
    #define c_realloc realloc
#endif


/* Use customized number representation -----------------------------------   */
#ifndef LONG
typedef int c_int;                   /* for indeces */
#else
typedef long c_int;                   /* for indeces */
#endif

#ifndef FLOAT // Doubles
typedef double c_float;              /* for numerical values  */
#define c_sqrt sqrt
#else         // Floats
typedef float c_float;                /* for numerical values  */
#define c_sqrt sqrtf
#endif




/* Use customized constants -----------------------------------------------   */
#ifndef OSQP_NULL
#define OSQP_NULL 0
#endif

#ifndef OSQP_NAN
#define OSQP_NAN ((c_float)0x7ff8000000000000)     // Not a Number
#endif

#ifndef OSQP_INFTY
#define OSQP_INFTY ((c_float)1e20)   // Infinity
#endif

/* Use customized operations */

#ifndef c_absval
#define c_absval(x) (((x) < 0) ? -(x) : (x))
#endif

#ifndef c_max
#define c_max(a, b) (((a) > (b)) ? (a) : (b))
#endif

#ifndef c_min
#define c_min(a, b) (((a) < (b)) ? (a) : (b))
#endif

/* Use customized functions -----------------------------------------------   */

#ifdef PRINTING
#ifdef MATLAB_MEX_FILE
#define c_print mexPrintf
// The following trick slows down the performance a lot. Since many solvers actually 
//call mexPrintf and immediately force print buffer flush
//otherwise messages don't appear until solver termination
//ugly because matlab does not provide a vprintf mex interface
// #include <stdarg.h>
// static int c_print(char *msg, ...)
// {
//   va_list argList;
//   va_start(argList, msg);
//   //message buffer
//   int bufferSize = 256;
//   char buffer[bufferSize];
//   vsnprintf(buffer,bufferSize-1, msg, argList);
//   va_end(argList);
//   int out = mexPrintf(buffer); //print to matlab display
//   mexEvalString("drawnow;");   // flush matlab print buffer
//   return out;
// }
#else
#define c_print printf
#endif
#endif


/* Use customized structures names */
typedef struct OSQP_PROBLEM_DATA Data;
typedef struct OSQP_POLISH Polish;
typedef struct OSQP_SETTINGS Settings;
typedef struct OSQP_SOLUTION Solution;
typedef struct OSQP_INFO Info;
typedef struct OSQP_SCALING Scaling;
typedef struct OSQP_WORK Work;
typedef struct OSQP_TIMER Timer;




#endif
