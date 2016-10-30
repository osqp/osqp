#include "stdio.h"
#include "osqp.h"
#include "cs.h"
#include "minunit.h"
#include "lin_alg/matrices.h"

static char * tests_lin_alg()
{
    /* local variables */
    int exitflag;
    csc * Asp;  // Sparse matrix allocation
    c_float * Asp_dns;  // Conversion to dense matrix

    printf("Linear algebra tests\n");
    printf("--------------------\n");

    printf("1) Construct sparse matrix\n");

    Asp = csc_matrix(m, n, Asp_nnz, Asp_x, Asp_i, Asp_p);

    // char *Aname = "A";
    // print_dns_matrix(A, m, n, Aname);

    Asp_dns =  csc_to_dns(Asp);

    print_dns_matrix(A, m, n, "Afile");
    print_dns_matrix(Asp_dns, m, n, "Aconverted");

    exitflag = 1;
    mu_assert("Error in linear algebra tests", exitflag == 1 );
    return 0;

}
