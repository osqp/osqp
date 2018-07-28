// Utilities for testing

#ifndef EMBEDDED

c_float* csc_to_dns(csc *M)
{
  c_int i, j = 0; // Predefine row index and column index
  c_int idx;

  // Initialize matrix of zeros
  c_float *A = (c_float *)c_calloc(M->m * M->n, sizeof(c_float));

  // Allocate elements
  for (idx = 0; idx < M->p[M->n]; idx++)
  {
    // Get row index i (starting from 1)
    i = M->i[idx];

    // Get column index j (increase if necessary) (starting from 1)
    while (M->p[j + 1] <= idx) {
      j++;
    }

    // Assign values to A
    A[j * (M->m) + i] = M->x[idx];
  }
  return A;
}


#endif // #ifndef EMBEDDED
