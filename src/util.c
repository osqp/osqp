#include "util.h"


/************************************
 * Printing Constants to set Layout *
 ************************************/
#if PRINTLEVEL > 1
#if PROFILING > 0
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
static const c_int LINE_LEN = 76;
#endif

/**********************
 * Utility Functions  *
 **********************/

#if PRINTLEVEL > 1

static void print_line(){
    for (c_int i = 0; i < LINE_LEN; ++i)
        c_print("-");
    c_print("\n");
}

void print_header(){
    c_print("%s ", HEADER[0]);
    for (c_int i=1; i < HEADER_LEN - 1; i++) c_print("  %s", HEADER[i]);
    c_print("%s\n", HEADER[HEADER_LEN - 1]);
}

void print_setup_header(const Data *data, const Settings *settings) {

    print_line();
    c_print("\tOSQP v%s - Operator Splitting QP Solver\n\t(c) ....., University of Oxford - Stanford University 2016\n", OSQP_VERSION);
    print_line();

    // Print variables and constraints
    c_print("Problem:  ");
    c_print("variables n = %i, constraints m = %i\n", data->n, data->m);

    // Print Settings
    // Print variables and constraints
    c_print("Settings: ");
    c_print("eps_abs = %.2e, eps_rel = %.2e,\n          rho = %.2f, alpha = %.2f, max_iter = %i\n",
            settings->eps_abs, settings->eps_rel, settings->rho, settings->alpha, settings->max_iter);
    if (settings->normalize)
        c_print("          scaling: active\n");
    else
        c_print("          scaling: inactive\n");
    if (settings->warm_start)
        c_print("          warm start: active\n");
    else
        c_print("          warm start: inactive\n");
    if (settings->polishing)
        c_print("          polishing: active\n");
    else
        c_print("          polishing: inactive\n");
    c_print("\n");

}


/* Print iteration summary */
void print_summary(Info * info){
    c_print("%*i ", (int)strlen(HEADER[0]), info->iter);
    c_print("%*.4e ", (int)HSPACE, info->obj_val);
    c_print("%*.4e ", (int)HSPACE, info->pri_res);
    c_print("%*.4e ", (int)HSPACE, info->dua_res);
    #if PROFILING > 0
    c_print("%*.2fs", 9, info->setup_time + info->solve_time);
    #endif
    c_print("\n");
}


/* Print polishing information */
void print_polishing(Info * info) {
    c_print("%*s ", (int)strlen(HEADER[0]), "PLSH");
    c_print("%*.4e ", (int)HSPACE, info->obj_val);
    c_print("%*.4e ", (int)HSPACE, info->pri_res);
    c_print("%*.4e ", (int)HSPACE, info->dua_res);
    #if PROFILING > 0
    c_print("%*.2fs", 9, info->setup_time + info->solve_time +
                         info->polish_time);
    #endif
    c_print("\n");
}


#endif /* End PRINTLEVEL > 1 */


#if PRINTLEVEL > 0
/* Print Footer */
void print_footer(Info * info, c_int polishing){

    #if PRINTLEVEL > 1
    c_print("\n"); // Add space after iterations
    #endif

    c_print("Status: %s\n", info->status);

    if (polishing && info->status_val == OSQP_SOLVED) {
        if (info->status_polish)
            c_print("Solution polishing: Successful\n");
        else
            c_print("Solution polishing: Unsuccessful\n");
    }

    c_print("Number of iterations: %i\n", info->iter);
    if (info->status_val == OSQP_SOLVED) {
        c_print("Optimal objective: %.4f\n", info->obj_val);
    }

    #if PROFILING > 0
    if (polishing && info->status_val == OSQP_SOLVED) {
        if (info->run_time > 1e-03) { // Time more than 1ms
            c_print("Timing: total  time = %.3fs\n        "
                    "setup  time = %.3fs\n        "
                    "solve  time = %.3fs\n        "
                    "polish time = %.3fs\n",
                    info->run_time, info->setup_time,
                    info->solve_time, info->polish_time);
        } else {
            c_print("Timing: total  time = %.3fms\n        "
                    "setup  time = %.3fms\n        "
                    "solve  time = %.3fms\n        "
                    "polish time = %.3fms\n",
                    1e03*info->run_time, 1e03*info->setup_time,
                    1e03*info->solve_time, 1e03*info->polish_time);
        }
    } else{
        if (info->run_time > 1e-03) { // Time more than 1ms
            c_print("Timing: total  time = %.3fs\n        "
                    "setup  time = %.3fs\n        "
                    "solve  time = %.3fs\n",
                    info->run_time, info->setup_time, info->solve_time);
        } else {
            c_print("Timing: total  time = %.3fms\n        "
                    "setup  time = %.3fms\n        "
                    "solve  time = %.3fms\n",
                    1e03*info->run_time, 1e03*info->setup_time,
                    1e03*info->solve_time);
        }
    }
    #endif

}

#endif


/* Set default settings from constants.h file */
/* assumes d->stgs already allocated memory */
void set_default_settings(Settings * settings) {
        settings->normalize = NORMALIZE; /* heuristic problem scaling */
        settings->rho = RHO; /* ADMM step */
        settings->max_iter = MAX_ITER; /* maximum iterations to take */
        settings->eps_abs = EPS_ABS;         /* absolute convergence tolerance */
        settings->eps_rel = EPS_REL;         /* relative convergence tolerance */
        settings->alpha = ALPHA;     /* relaxation parameter */
        settings->delta = DELTA;    /* regularization parameter for polishing */
        settings->polishing = POLISHING;     /* ADMM solution polishing: 1 */
        settings->verbose = VERBOSE;     /* x equality constraint scaling: 1e-3 */
        settings->warm_start = WARM_START;     /* x equality constraint scaling: 1e-3 */

}


/* Copy settings creating a new settings structure */
Settings * copy_settings(Settings * settings){
    Settings * new = c_malloc(sizeof(Settings));

    // Copy settings
    new->normalize = settings->normalize;
    new->rho = settings->rho;
    new->max_iter = settings->max_iter;
    new->eps_abs = settings->eps_abs;
    new->eps_rel = settings->eps_rel;
    new->alpha = settings->alpha;
    new->delta = settings->delta;
    new->polishing = settings->polishing;
    new->verbose = settings->verbose;
    new->warm_start = settings->warm_start;

    return new;
}




/*******************
* Timer Functions *
*******************/

#if PROFILING > 0

// Windows
#if (defined WIN32 || _WIN64)

void tic(Timer* t)
{
        QueryPerformanceFrequency(&t->freq);
        QueryPerformanceCounter(&t->tic);
}

c_float toc(Timer* t)
{
        QueryPerformanceCounter(&t->toc);
        return ((t->toc.QuadPart - t->tic.QuadPart) / (pfloat)t->freq.QuadPart);
}

// Mac
#elif (defined __APPLE__)

void tic(Timer* t)
{
        /* read current clock cycles */
        t->tic = mach_absolute_time();
}

c_float toc(Timer* t)
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
void tic(Timer* t)
{
        clock_gettime(CLOCK_MONOTONIC, &t->tic);
}


/* return time passed since last call to tic on this timer */
c_float toc(Timer* t)
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







/* ================================= DEBUG FUNCTIONS ======================= */

/* Convert sparse CSC to dense */
c_float * csc_to_dns(csc * M)
{
        c_int i, j=0; // Predefine row index and column index

        // Initialize matrix of zeros
        c_float * A = (c_float *)c_calloc(M->m * M->n, sizeof(c_float));

        // Allocate elements
        for (c_int idx = 0; idx < M->nzmax; idx++)
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

#if PRINTLEVEL > 2

/* Print a csc sparse matrix */
void print_csc_matrix(csc* M, char * name)
{
        c_print("%s :\n", name);
        c_int j, i, row_start, row_stop;
        c_int k = 0;
        for(j=0; j<M->n; j++) {
                row_start = M->p[j];
                row_stop = M->p[j+1];
                if (row_start == row_stop)
                        continue;
                else {
                        for(i=row_start; i<row_stop; i++ ) {
                                c_print("\t[%3u,%3u] = %g\n", M->i[i], j, M->x[k++]);
                        }
                }
        }
}

/* Print a triplet format sparse matrix */
void print_trip_matrix(csc* M, char * name)
{
        c_print("%s :\n", name);
        c_int k = 0;
        for (k=0; k<M->nz; k++){
            c_print("\t[%3u, %3u] = %g\n", M->i[k], M->p[k], M->x[k]);
        }
}


/* Print a dense matrix */
void print_dns_matrix(c_float * M, c_int m, c_int n, char *name)
{
        c_print("%s = \n\t", name);
        for(c_int i=0; i<m; i++) { // Cycle over rows
                for(c_int j=0; j<n; j++) { // Cycle over columns
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
void print_vec(c_float * v, c_int n, char *name){
        print_dns_matrix(v, 1, n, name);
}


// Print int array
void print_vec_int(c_int * x, c_int n, char *name) {
    c_print("%s = [", name);
    for(c_int i=0; i<n; i++) {
        c_print(" %d ", x[i]);
    }
    c_print("]\n");
}


#endif
