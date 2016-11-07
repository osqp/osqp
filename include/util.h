#ifndef UTIL_H
#define UTIL_H

#include "constants.h"
#include "cs.h"
#include "osqp.h"

/**********************
 * Utility Functions  *
 **********************/

/* Set default settings from constants.h file
 * assumes settings already allocated inmemory
*/
void set_default_settings(Settings * settings);

/* Copy settings creating a new settings structure */
Settings * copy_settings(Settings * settings);

#if PRINTLEVEL > 1
/* Print Header before running the algorithm */
void print_setup_header(const Data *data, const Settings *settings);

/* Print Header with data to be displayed per iteration */
void print_header();

/* Print iteration summary */
void print_summary(Info * info);

#endif

#if PRINTLEVEL > 0

/* Print Footer */
void print_footer(Info * info);

#endif


/*********************************
 * Timer Structs and Functions * *
 *********************************/

#if PROFILING > 0

// Windows
#if (defined _WIN32 || defined _WIN64 || defined _WINDLL )

#include <windows.h>

typedef struct timer{
	LARGE_INTEGER tic;
	LARGE_INTEGER toc;
	LARGE_INTEGER freq;
} timer;

// Mac
#elif (defined __APPLE__)

#include <mach/mach_time.h>

/* Use MAC OSX  mach_time for timing */
typedef struct timer{
	uint64_t tic;
	uint64_t toc;
	mach_timebase_info_data_t tinfo;
} timer;

// Linux
#else

/* Use POSIX clocl_gettime() for timing on non-Windows machines */
#include <time.h>
#include <sys/time.h>

typedef struct timer{
	struct timespec tic;
	struct timespec toc;
} timer;

#endif

/**
 * Timer Methods
 */
void tic(timer* t);
c_float toc(timer* t);

#endif /* END IF PROFILING > 0 */



/* ================================= DEBUG FUNCTIONS ======================= */
// TODO: Remove debug functions!

/* Compare CSC matrices */
c_int is_eq_csc(csc *A, csc *B, c_float tol);

/* Convert sparse CSC to dense */
c_float * csc_to_dns(csc * M);

#if PRINTLEVEL > 2
#include <stdio.h>


/* Print a csc sparse matrix */
void print_csc_matrix(csc* M, char * name);

/* Print a triplet format sparse matrix */
void print_trip_matrix(csc* M, char * name);

/* Print a dense matrix */
void print_dns_matrix(c_float * M, c_int m, c_int n, char *name);

/* Print vector  */
void print_vec(c_float * V, c_int n, char *name);

#endif



#endif
