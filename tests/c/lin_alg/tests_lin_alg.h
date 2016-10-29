#include "stdio.h"
#include "osqp.h"
#include "cs.h"
#include "minunit.h"
#include "lin_alg/matrices.h"

static char * tests_lin_alg()
{
    /* local variables */
    int exitflag;

    // /* local variables */
    // pwork *mywork;
    // idxint exitflag;
    //
    // /* set up data */
	// mywork = ECOS_setup(MPC01_n, MPC01_m, MPC01_p, MPC01_l, MPC01_ncones, MPC01_q, 0,
    //                     MPC01_Gpr, MPC01_Gjc, MPC01_Gir,
    //                     MPC01_Apr, MPC01_Ajc, MPC01_Air,
    //                     MPC01_c, MPC01_h, MPC01_b);
    // if( mywork != NULL ){
    //
    //     /* solve */
    //     exitflag = ECOS_solve(mywork); }
    //
    // else exitflag = ECOS_FATAL;
    //
    // /* clean up memory */
    // ECOS_cleanup(mywork, 0);


    printf("Linear algebra tests...\n");

    char *Aname = "A";

    print_dns_matrix(A, m, n, Aname);

    exitflag = 1;
    mu_assert("Error in linear algebra tests", exitflag == 1 );
    return 0;

}
