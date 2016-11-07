#include "util.h"


/************************************
 * Printing Constants to set Layout *
 ************************************/
static const char *HEADER[] = {
 "Iter ",   " Obj  Val ",  "  Pri  Res ", "    Dua  Res "
};
static const c_int HSPACE = 12;
static const c_int HEADER_LEN = 4;
static const c_int LINE_LEN = 76;

/* ================================= PRINTING FUNCTIONS ==================== */

static void print_line(){
    for (c_int i = 0; i < LINE_LEN; ++i)
        c_print("-");
    c_print("\n");
}

void print_header(){
    c_print("%s| ", HEADER[0]);
    for (c_int i=1; i < HEADER_LEN - 1; i++) c_print("  %s", HEADER[i]);
    c_print("%s\n", HEADER[HEADER_LEN - 1]);
}

void print_setup_header(const Data *data, const Settings *settings) {

    print_line();
    c_print("\tOSQP v%s - Operator Splitting QP Solver\n\t(c) ....., University of Oxford - Stanford University 2016\n", OSQP_VERSION);
    print_line();

    // Print variables and constraints
    c_print("Problem:  ");
    c_print("variables n = %i, constraints m = %i\n\n", data->n, data->m);

    // Print Settings
    // Print variables and constraints
    c_print("Settings: ");
    c_print("eps_abs = %.2e, eps_rel = %.2e,\n          rho = %.2f, alpha = %.2f, max_iter = %i\n",
            settings->eps_abs, settings->eps_rel, settings->rho, settings->alpha, settings->max_iter);
    if (settings->normalize)
        c_print("          scaling: active\n");
    else
        c_print("          scaling: inactive\n");
    c_print("\n");

}


/* Print iteration summary */
void print_summary(Info * info){
    c_print("%*.i| ", (int)strlen(HEADER[0]), info->iter);
    c_print("%*.4e ", (int)HSPACE, info->obj_val);
    c_print("%*.4e ", (int)HSPACE, info->pri_res);
    c_print("%*.4e ", (int)HSPACE, info->dua_res);
    c_print("\n");
}








/* ================================= OTHER FUNCTIONS ======================= */

/* Set default settings from constants.h file */
/* assumes d->stgs already allocated memory */
void set_default_settings(Settings * settings) {
        settings->normalize = NORMALIZE; /* heuristic problem scaling */
        settings->max_iter = MAX_ITER; /* maximum iterations to take */
        settings->eps_abs = EPS_ABS;         /* absolute convergence tolerance */
        settings->eps_rel = EPS_REL;         /* relative convergence tolerance */
        settings->alpha = ALPHA;     /* relaxation parameter */
        settings->verbose = VERBOSE;     /* x equality constraint scaling: 1e-3 */
        settings->warm_start = WARM_START;     /* x equality constraint scaling: 1e-3 */
}


/* Copy settings creating a new settings structure */
Settings * copy_settings(Settings * settings){
    Settings * new = c_malloc(sizeof(Settings));

    // Copy settings
    new->normalize = settings->normalize;
    new->max_iter = settings->max_iter;
    new->eps_abs = settings->eps_abs;
    new->eps_rel = settings->eps_rel;
    new->alpha = settings->alpha;
    new->verbose = settings->verbose;
    new->warm_start = settings->warm_start;

    return new;
}

/* ================================= DEBUG FUNCTIONS ======================= */
#if PRINTLEVEL > 2

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
c_int is_eq_csc(csc *A, csc *B){
        c_int j, i;
        // If number of columns does not coincide, they are not equal.
        if (A->n != B->n) return 0;

        for (j=0; j<A->n; j++) { // Cycle over columns j

                // if column pointer does not coincide, they are not equal
                if (A->p[j] != B->p[j]) return 0;

                for (i=A->p[j]; i<A->p[j+1]; i++) { // Cycle rows i in column j
                        if (A->i[i] != B->i[i] || // Different row indices
                            c_abs(A->x[i] - B->x[i]) > TESTS_TOL) {
                                return 0;
                        }
                }
        }
        return(1);
}


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
void print_vec(c_float * V, c_int n, char *name){
        print_dns_matrix(V, 1, n, name);
}

#endif
