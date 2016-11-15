#ifndef CHAIN80W_H
#define CHAIN80W_H
#include "osqp.h"


/* function to generate problem data structure */
Data * generate_problem_chain80w();

/* function to clean problem data structure */
c_int clean_problem_chain80w(Data * data);

#endif
