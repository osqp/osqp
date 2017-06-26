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
#define HEADER_LINE_LEN 60
#endif

/**********************
 * Utility Functions  *
 **********************/


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
    c_print("        OSQP v%s  -  Operator Splitting QP Solver\n"
            "           (c) Bartolomeo Stellato,  Goran Banjac\n"
            "     University of Oxford  -  Stanford University 2017\n",
            OSQP_VERSION);
    print_line();

    // Print variables and constraints
    c_print("Problem:  ");
    c_print("variables n = %i, constraints m = %i\n", (int)data->n, (int)data->m);

    // Print Settings
    // Print variables and constraints
    c_print("Settings: ");
    c_print("eps_abs = %.1e, eps_rel = %.1e,\n          ", settings->eps_abs, settings->eps_rel);
    c_print("eps_prim_inf = %.1e, eps_dual_inf = %.1e,\n          ", settings->eps_prim_inf, settings->eps_dual_inf);
    c_print("rho = %.2e ", settings->rho);
    if (settings->auto_rho) c_print("(auto)");
    c_print("\n          ");
    c_print("sigma = %.1e, alpha = %.1e, \n          ", settings->sigma, settings->alpha);
    c_print("max_iter = %i\n", (int)settings->max_iter);

    if (settings->early_terminate)
        c_print("          early_terminate: on (interval %i)\n", (int)settings->early_terminate_interval);
    else
        c_print("          early_terminate: off \n");
    if (settings->scaling)
        c_print("          scaling: on, ");
    else
        c_print("          scaling: off, ");
    if (settings->scaled_termination)
        c_print("scaled_termination: on\n");
    else
        c_print("scaled_termination: off\n");
    if (settings->warm_start)
        c_print("          warm start: on, ");
    else
        c_print("          warm start: off, ");
    if (settings->polish)
        c_print("polish: on\n");
    else
        c_print("polish: off\n");
    c_print("\n");
}


void print_summary(OSQPWorkspace * work){
    OSQPInfo * info;
    info = work->info;

    c_print("%*i ", (int)strlen(HEADER[0]), (int)info->iter);
    c_print("%*.4e ", (int)HSPACE, info->obj_val);
    c_print("%*.4e ", (int)HSPACE, info->pri_res);
    c_print("%*.4e ", (int)HSPACE, info->dua_res);
    #ifdef PROFILING
    c_print("%*.2fs", 9, info->setup_time + info->solve_time);
    #endif
    c_print("\n");

    work->summary_printed = 1; // Summary has been printed
}


void print_polish(OSQPWorkspace * work) {
    OSQPInfo * info;
    info = work->info;
    
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

void print_footer(OSQPInfo * info, c_int polish){

    #ifdef PRINTING
    c_print("\n"); // Add space after iterations
    #endif

    c_print("Status: %s\n", info->status);

    if (polish && info->status_val == OSQP_SOLVED) {
        if (info->status_polish == 1){
            c_print("Solution polish: Successful\n");
        } else if (info->status_polish < 0){
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


void set_default_settings(OSQPSettings * settings) {
        settings->scaling = SCALING; /* heuristic problem scaling */

        #if EMBEDDED != 1
        settings->scaling_iter = SCALING_ITER;
        #endif

        settings->rho = (c_float) RHO; /* ADMM step */
        settings->sigma = (c_float) SIGMA; /* ADMM step */
        settings->max_iter = MAX_ITER; /* maximum iterations to take */
        settings->eps_abs = (c_float) EPS_ABS;         /* absolute convergence tolerance */
        settings->eps_rel = (c_float) EPS_REL;         /* relative convergence tolerance */
        settings->eps_prim_inf = (c_float) EPS_PRIM_INF;         /* primal infeasibility tolerance */
        settings->eps_dual_inf = (c_float) EPS_DUAL_INF;         /* dual infeasibility tolerance */
        settings->alpha = (c_float) ALPHA;     /* relaxation parameter */

        #ifndef EMBEDDED
        settings->delta = DELTA;    /* regularization parameter for polish */
        settings->polish = POLISH;     /* ADMM solution polish: 1 */
        settings->pol_refine_iter = POL_REFINE_ITER; /* iterative refinement
                                                        steps in polish */
        settings->auto_rho = AUTO_RHO; /* automatic rho computation */
        settings->verbose = VERBOSE;     /* print output */
        #endif

        settings->scaled_termination = SCALED_TERMINATION;     /* Evaluate scaled termination criteria*/
        settings->early_terminate = EARLY_TERMINATE;     /* Evaluate termination criteria */
        settings->early_terminate_interval = EARLY_TERMINATE_INTERVAL;     /* Evaluate termination at certain interval */
        settings->warm_start = WARM_START;     /* x equality constraint scaling: 1e-3 */

}

#ifndef EMBEDDED

OSQPSettings * copy_settings(OSQPSettings * settings){
    OSQPSettings * new = c_malloc(sizeof(OSQPSettings));

    // Copy settings
    new->scaling = settings->scaling;
    new->scaling_iter = settings->scaling_iter;
    new->rho = settings->rho;
    new->sigma = settings->sigma;
    new->max_iter = settings->max_iter;
    new->eps_abs = settings->eps_abs;
    new->eps_rel = settings->eps_rel;
    new->eps_prim_inf = settings->eps_prim_inf;
    new->eps_dual_inf = settings->eps_dual_inf;
    new->alpha = settings->alpha;
    new->delta = settings->delta;
    new->polish = settings->polish;
    new->pol_refine_iter = settings->pol_refine_iter;
    new->auto_rho = settings->auto_rho;
    new->verbose = settings->verbose;
    new->scaled_termination = settings->scaled_termination;
    new->early_terminate = settings->early_terminate;
    new->early_terminate_interval = settings->early_terminate_interval;
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


void print_trip_matrix(csc* M, const char * name)
{
        c_int k = 0;

        // Print name
        c_print("%s :\n", name);

        for (k=0; k<M->nz; k++){
            c_print("\t[%3u, %3u] = %g\n", (int)M->i[k], (int)M->p[k], M->x[k]);
        }
}



void print_dns_matrix(c_float * M, c_int m, c_int n, const char *name)
{
        c_int i, j;
        c_print("%s : \n\t", name);
        for(i=0; i<m; i++) { // Cycle over rows
                for(j=0; j<n; j++) { // Cycle over columns
                        if (j < n - 1)
                                // c_print("% 14.12e,  ", M[j*m+i]);
                                c_print("% .8f,  ", M[j*m+i]);

                        else
                                // c_print("% 14.12e;  ", M[j*m+i]);
                                c_print("% .8f;  ", M[j*m+i]);
                }
                if (i < m - 1) {
                        c_print("\n\t");
                }
        }
        c_print("\n");
}


void print_vec(c_float * v, c_int n, const char *name){
        print_dns_matrix(v, 1, n, name);
}



void print_vec_int(c_int * x, c_int n, const char *name) {
    c_int i;
    c_print("%s = [", name);
    for(i=0; i<n; i++) {
        c_print(" %d ", (int)x[i]);
    }
    c_print("]\n");
}


#endif  // PRINTING
