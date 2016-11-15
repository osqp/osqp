#ifndef DIESEL_H
#define DIESEL_H
#include "osqp.h"


/* function to generate problem data structure */
Data * generate_problem_diesel();

/* function to clean problem data structure */
c_int clean_problem_diesel(Data * data);

#endif
