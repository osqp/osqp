#include "util.h"

/***************
 * Versioning  *
 ***************/
const char *osqp_version(void) {
    return OSQP_VERSION;
}



/************************************
 * Printing Constants to set Layout *
 ************************************/
#ifdef PRINTING
#ifdef PROFILING
static const char *HEADER[] = {
 "Iter",   " Obj  Val ",  "  Pri  Res ", "  Dua  Res ", "      Time "
};
static const c_int HEADER_LEN = 5;
#else
static const char *HEADER[] = {
 "Iter",   " Obj  Val ",  "  Pri  Res ", "    Dua  Res "
};
static const c_int HEADER_LEN = 4;
#endif
static const c_int HSPACE = 12;
#define HEADER_LINE_LEN 55
#endif

/**********************
 * Utility Functions  *
 **********************/


 /* Custom string copy to avoid string.h library */
void c_strcpy(char dest[], const char source[]){
int i = 0;
    while (1) {
       dest[i] = source[i];
       if (dest[i] == '\0') break;
       i++;
 } }


#ifdef PRINTING

static void print_line(void){
    char theLine[HEADER_LINE_LEN+1];
    c_int i;
    for (i = 0; i < HEADER_LINE_LEN; ++i)
        theLine[i] = '-';
    theLine[HEADER_LINE_LEN] = '\0';
    c_print("%s\n",theLine);
}

void print_header(void){
    c_int i;
    c_print("%s ", HEADER[0]);
    for (i=1; i < HEADER_LEN - 1; i++) c_print("  %s", HEADER[i]);
    c_print("%s\n", HEADER[HEADER_LEN - 1]);
}

void print_setup_header(const OSQPData *data, const OSQPSettings *settings) {
    print_line();
    c_print("      OSQP v%s  -  Operator Splitting QP Solver\n"
            "     (c) .....,\n"
            "   University of Oxford  -  Stanford University 2016\n",
            OSQP_VERSION);
    print_line();

    // Print variables and constraints
    c_print("Problem:  ");
    c_print("variables n = %i, constraints m = %i\n", (int)data->n, (int)data->m);

    // Print Settings
    // Print variables and constraints
    c_print("Settings: ");
    c_print("eps_abs = %.2e, eps_rel = %.2e,\n          "
            "eps_inf = %.2e, eps_unb = %.2e,\n          "
            "rho = %.2f, sigma = %.2f, alpha = %.2f, \n          max_iter = %i\n",
            settings->eps_abs, settings->eps_rel, settings->eps_inf, settings->eps_unb, settings->rho, settings->sigma,
            settings->alpha, (int)settings->max_iter);
    if (settings->scaling)
        c_print("          scaling: active\n");
    else
        c_print("          scaling: inactive\n");
    if (settings->warm_start)
        c_print("          warm start: active\n");
    else
        c_print("          warm start: inactive\n");
    if (settings->polish)
        c_print("          polish: active\n");
    else
        c_print("          polish: inactive\n");
    c_print("\n");
}


/* Print iteration summary */
void print_summary(OSQPInfo * info){
    c_print("%*i ", (int)strlen(HEADER[0]), (int)info->iter);
    c_print("%*.4e ", (int)HSPACE, info->obj_val);
    c_print("%*.4e ", (int)HSPACE, info->pri_res);
    c_print("%*.4e ", (int)HSPACE, info->dua_res);
    #ifdef PROFILING
    c_print("%*.2fs", 9, info->setup_time + info->solve_time);
    #endif
    c_print("\n");
}


/* Print polish information */
void print_polish(OSQPInfo * info) {
    c_print("%*s ", (int)strlen(HEADER[0]), "PLSH");
    c_print("%*.4e ", (int)HSPACE, info->obj_val);
    c_print("%*.4e ", (int)HSPACE, info->pri_res);
    c_print("%*.4e ", (int)HSPACE, info->dua_res);
    #ifdef PROFILING
    c_print("%*.2fs", 9, info->setup_time + info->solve_time +
                         info->polish_time);
    #endif
    c_print("\n");
}


#endif /* End #ifdef PRINTING */


#ifdef PRINTING
/* Print Footer */
void print_footer(OSQPInfo * info, c_int polish){

    #ifdef PRINTING
    c_print("\n"); // Add space after iterations
    #endif

    c_print("Status: %s\n", info->status);

    if (polish && info->status_val == OSQP_SOLVED) {
        if (info->status_polish == 1){
            c_print("Solution polish: Successful\n");
        } else if (info->status_polish == -1){
            c_print("Solution polish: Unsuccessful\n");
        }
    }

    c_print("Number of iterations: %i\n", (int)info->iter);
    if (info->status_val == OSQP_SOLVED) {
        c_print("Optimal objective: %.4f\n", info->obj_val);
    }

    #ifdef PROFILING
    if (info->run_time > 1e-03) { // Time more than 1ms
        c_print("Run time: %.3fs\n", info->run_time);
    } else {
        c_print("Run time: %.3fms\n", 1e03*info->run_time);
    }
    #endif
    c_print("\n");

}

#endif


/* Set default settings from constants.h file */
/* assumes d->stgs already allocated memory */
void set_default_settings(OSQPSettings * settings) {
        settings->scaling = SCALING; /* heuristic problem scaling */

        #if EMBEDDED != 1
        settings->scaling_norm = SCALING_NORM;
        settings->scaling_iter = SCALING_ITER;
        #endif

        settings->rho = RHO; /* ADMM step */
        settings->sigma = SIGMA; /* ADMM step */
        settings->max_iter = MAX_ITER; /* maximum iterations to take */
        settings->eps_abs = EPS_ABS;         /* absolute convergence tolerance */
        settings->eps_rel = EPS_REL;         /* relative convergence tolerance */
        settings->eps_inf = EPS_INF;         /* infeasibility tolerance */
        settings->eps_unb = EPS_UNB;         /* unboundedness tolerance */
        settings->alpha = ALPHA;     /* relaxation parameter */

        #ifndef EMBEDDED
        settings->delta = DELTA;    /* regularization parameter for polish */
        settings->polish = POLISH;     /* ADMM solution polish: 1 */
        settings->pol_refine_iter = POL_REFINE_ITER; /* iterative refinement
                                                        steps in polish */
        settings->verbose = VERBOSE;     /* print output */
        #endif

        settings->early_terminate = EARLY_TERMINATE;     /* Evaluate termination criteria */
        settings->warm_start = WARM_START;     /* x equality constraint scaling: 1e-3 */

}

#ifndef EMBEDDED

/* Copy settings creating a new settings structure */
OSQPSettings * copy_settings(OSQPSettings * settings){
    OSQPSettings * new = c_malloc(sizeof(OSQPSettings));

    // Copy settings
    new->scaling = settings->scaling;
    new->scaling_norm = settings->scaling_norm;
    new->scaling_iter = settings->scaling_iter;
    new->rho = settings->rho;
    new->sigma = settings->sigma;
    new->max_iter = settings->max_iter;
    new->eps_abs = settings->eps_abs;
    new->eps_rel = settings->eps_rel;
    new->eps_inf = settings->eps_inf;
    new->eps_unb = settings->eps_unb;
    new->alpha = settings->alpha;
    new->delta = settings->delta;
    new->polish = settings->polish;
    new->pol_refine_iter = settings->pol_refine_iter;
    new->verbose = settings->verbose;
    new->early_terminate = settings->early_terminate;
    new->warm_start = settings->warm_start;

    return new;
}

#endif  // #ifndef EMBEDDED



/*******************
* Timer Functions *
*******************/

#ifdef PROFILING

// Windows
#if IS_WINDOWS

void tic(OSQPTimer* t)
{
        QueryPerformanceFrequency(&t->freq);
        QueryPerformanceCounter(&t->tic);
}

c_float toc(OSQPTimer* t)
{
        QueryPerformanceCounter(&t->toc);
        return ((t->toc.QuadPart - t->tic.QuadPart) / (c_float)t->freq.QuadPart);
}

// Mac
#elif IS_MAC

void tic(OSQPTimer* t)
{
        /* read current clock cycles */
        t->tic = mach_absolute_time();
}

c_float toc(OSQPTimer* t)
{

        uint64_t duration; /* elapsed time in clock cycles*/

        t->toc = mach_absolute_time();
        duration = t->toc - t->tic;

        /*conversion from clock cycles to nanoseconds*/
        mach_timebase_info(&(t->tinfo));
        duration *= t->tinfo.numer;
        duration /= t->tinfo.denom;

        return (c_float)duration / 1e9;
}


// Linux
#else

/* read current time */
void tic(OSQPTimer* t)
{
        clock_gettime(CLOCK_MONOTONIC, &t->tic);
}


/* return time passed since last call to tic on this timer */
c_float toc(OSQPTimer* t)
{
        struct timespec temp;

        clock_gettime(CLOCK_MONOTONIC, &t->toc);

        if ((t->toc.tv_nsec - t->tic.tv_nsec)<0) {
                temp.tv_sec = t->toc.tv_sec - t->tic.tv_sec-1;
                temp.tv_nsec = 1e9+t->toc.tv_nsec - t->tic.tv_nsec;
        } else {
                temp.tv_sec = t->toc.tv_sec - t->tic.tv_sec;
                temp.tv_nsec = t->toc.tv_nsec - t->tic.tv_nsec;
        }
        return (c_float)temp.tv_sec + (c_float)temp.tv_nsec / 1e9;
}

#endif

#endif // If Profiling end






/* ==================== DEBUG FUNCTIONS ======================= */

#ifndef EMBEDDED
/* Convert sparse CSC to dense */
c_float * csc_to_dns(csc * M)
{
        c_int i, j=0; // Predefine row index and column index
        c_int idx;

        // Initialize matrix of zeros
        c_float * A = (c_float *)c_calloc(M->m * M->n, sizeof(c_float));

        // Allocate elements
        for (idx = 0; idx < M->p[M->n]; idx++)
        {
                // Get row index i (starting from 1)
                i = M->i[idx];

                // Get column index j (increase if necessary) (starting from 1)
                while (M->p[j+1] <= idx) {
                        j++;
                }

                // Assign values to A
                A[j*(M->m)+i] = M->x[idx];

        }
        return A;
}


/* Compare sparse matrices */
c_int is_eq_csc(csc *A, csc *B, c_float tol){
        c_int j, i;
        // If number of columns does not coincide, they are not equal.
        if (A->n != B->n) return 0;

        for (j=0; j<A->n; j++) { // Cycle over columns j

                // if column pointer does not coincide, they are not equal
                if (A->p[j] != B->p[j]) return 0;

                for (i=A->p[j]; i<A->p[j+1]; i++) { // Cycle rows i in column j
                        if (A->i[i] != B->i[i] || // Different row indices
                            c_absval(A->x[i] - B->x[i]) > tol) {
                                return 0;
                        }
                }
        }
        return(1);
}

#endif  // #ifndef EMBEDDED


#ifdef PRINTING
/* Print a csc sparse matrix */
void print_csc_matrix(csc* M, const char * name)
{
        c_int j, i, row_start, row_stop;
        c_int k=0;

        // Print name
        c_print("%s :\n", name);

        for(j=0; j<M->n; j++) {
                row_start = M->p[j];
                row_stop = M->p[j+1];
                if (row_start == row_stop)
                        continue;
                else {
                        for(i=row_start; i<row_stop; i++ ) {
                                c_print("\t[%3u,%3u] = %g\n", (int)M->i[i], (int)j, M->x[k++]);
                        }
                }
        }
}

/* Print a triplet format sparse matrix */
void print_trip_matrix(csc* M, const char * name)
{
        c_int k = 0;

        // Print name
        c_print("%s :\n", name);

        for (k=0; k<M->nz; k++){
            c_print("\t[%3u, %3u] = %g\n", (int)M->i[k], (int)M->p[k], M->x[k]);
        }
}


/* Print a dense matrix */
void print_dns_matrix(c_float * M, c_int m, c_int n, const char *name)
{
        c_int i, j;
        c_print("%s : \n\t", name);
        for(i=0; i<m; i++) { // Cycle over rows
                for(j=0; j<n; j++) { // Cycle over columns
                        if (j < n - 1)
                                // c_print("% 14.12e,  ", M[j*m+i]);
                                c_print("% .5f,  ", M[j*m+i]);

                        else
                                // c_print("% 14.12e;  ", M[j*m+i]);
                                c_print("% .5f;  ", M[j*m+i]);
                }
                if (i < m - 1) {
                        c_print("\n\t");
                }
        }
        c_print("\n");
}

/* Print vector */
void print_vec(c_float * v, c_int n, const char *name){
        print_dns_matrix(v, 1, n, name);
}


// Print int array
void print_vec_int(c_int * x, c_int n, const char *name) {
    c_int i;
    c_print("%s = [", name);
    for(i=0; i<n; i++) {
        c_print(" %d ", (int)x[i]);
    }
    c_print("]\n");
}


#endif  // PRINTING
