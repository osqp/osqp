#ifndef NORMALIZE_H
#define NORMALIZE_H

// Functions to scale problem data

#include "osqp.h"


// Scale data stored in workspace
c_int scale_data(Work * work);

// Scale solution
// c_int scale_solution(Work *work);

// Unscale solution
c_int unscale_solution(Work *work);


#endif
