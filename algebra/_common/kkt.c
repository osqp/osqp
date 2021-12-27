#include "kkt.h"


//add an offset to every term in the upper nxn block.
//assumes triu CSC or CSR format, with fully populated diagonal.
//format = 0 / CSC:  diagonal terms are last in every column.
//format = 1 / CSR:  diagonal terms are first in every row.
static void _kkt_shifts_param1(csc* KKT, c_float param1, c_int n, c_int format){
  int i;
  int offset = format == 0 ? 1 : 0;
  for(i = 0; i < n; i++){ KKT->x[KKT->p[i+offset]-offset] += param1;}
  return;
}

//*subtract* an offset to every term in the lower mxm block.
//assumes triu CSC P/A formats, with fully populated diagonal.
//KKT format = 0 / CSC:  diagonal terms are last in every column.
//KKT format = 1 / CSR:  diagonal terms are first in every row.
static void _kkt_shifts_param2(csc* KKT, c_float* param2, c_float param2_sc, c_int startcol, c_int blockwidth, c_int format){

  int i;
  int offset = format == 0 ? 1 : 0;

  if(param2){
    for(i = 0; i < blockwidth; i++){ KKT->x[KKT->p[i + startcol + offset]-offset] -= param2[i];}
  }else{
    for(i = 0; i < blockwidth; i++){ KKT->x[KKT->p[i + startcol + offset]-offset] -= param2_sc;}
  }
}

#ifndef EMBEDDED

//the remainder of the private functions here are for KKT
//assembly ONLY, so don't included them when EMBEDDED

//increment the K colptr by the number of nonzeros
//in a square diagonal matrix placed on the diagonal.
//Used to increment, e.g. the lower RHS block diagonal
static void _kkt_colcount_diag(csc* K, c_int initcol, c_int blockcols){

    c_int j;
    for(j = initcol; j < (initcol + blockcols); j++){
        K->p[j]++;
    }
    return;
}

//same as _kkt_colcount_diag, but counts places
//where the input matrix M has a missing
//diagonal entry.  M must be square and TRIU
static void _kkt_colcount_missing_diag(csc* K, csc* M, c_int initcol){

    c_int j;
    for (j = 0; j < M->n; j++){
        //if empty column or last entry not on diagonal..
        if((M->p[j] == M->p[j+1]) || (M->i[M->p[j+1]-1] != j)) {
            K->p[j + initcol]++;
        }
    }
    return;
}

//increment K colptr by the number of nonzeros in M
static void _kkt_colcount_block(csc* K, csc* M, c_int initcol, c_int istranspose){

    c_int nnzM, j;

    if(istranspose){
      nnzM = M->p[M->n];
      for (j = 0; j < nnzM; j++){
        K->p[M->i[j] + initcol]++;
      }
    }
    else {
      //just add the column count
      for (j = 0; j < M->n; j++){
          K->p[j + initcol] += M->p[j+1] - M->p[j];
      }
    }
    return;
}


//populate values from M using the K colptr as indicator of
//next fill location in each row
static void _kkt_fill_block(
  csc* K, csc* M,
  c_int* MtoKKT,
  c_int initrow,
  c_int initcol,
  c_int istranspose)
{
    c_int ii, jj, row, col, dest;

    for(ii=0; ii < M->n; ii++){
        for(jj = M->p[ii]; jj < M->p[ii+1]; jj++){
            if(istranspose){
                col = M->i[jj] + initcol;
                row = ii + initrow;
            }
            else {
                col = ii + initcol;
                row = M->i[jj] + initrow;
            }

            dest       = K->p[col]++;
            K->i[dest] = row;
            K->x[dest] = M->x[jj];
            if(MtoKKT != OSQP_NULL){MtoKKT[jj] = dest;}
        }
    }
    return;
}

//increment the K colptr by the number of elements
//in a square diagonal matrix placed on the diagonal.
//Used to increment, e.g. the lower RHS block diagonal.
//values are filled with structural zero
static void _kkt_fill_diag_zeros(csc* K,c_int* rhotoKKT, c_int offset, c_int blockdim){

    c_int j, dest, col;
    for(j = 0; j < blockdim; j++){
        col         = j + offset;
        dest        = K->p[col];
        K->i[dest]  = col;
        K->x[dest]  = 0.0;  //structural zero
        K->p[col]++;
        if(rhotoKKT != OSQP_NULL){rhotoKKT[j] = dest;}
    }
    return;
}

//same as _kkt_fill_diag_zeros, but only places
//entries where the input matrix M has a missing
//diagonal entry.  M must be square and TRIU
static void _kkt_fill_missing_diag_zeros(csc* K,csc* M, c_int offset){

    c_int j, dest;
    for(j = 0; j < M->n; j++){
        //fill out missing diagonal terms only:
        //if completely empty column last element is not on diagonal..
        if((M->p[j] == M->p[j+1]) ||
           (M->i[M->p[j+1]-1] != j))
        {
            dest           = K->p[j + offset];
            K->i[dest]  = j + offset;
            K->x[dest]  = 0.0;  //structural zero
            K->p[j]++;;
        }
    }
    return;
}

static void _kkt_colcount_to_colptr(csc* K){

    c_int j, count;
    c_int currentptr = 0;

    for(j = 0; j <= K->n; j++){
       count        = K->p[j];
       K->p[j]      = currentptr;
       currentptr  += count;
    }
    return;
}

static void _kkt_backshift_colptrs(csc* K){

    int j;
    for(j = K->n; j > 0; j--){
        K->p[j] = K->p[j-1];
    }
    K->p[0] = 0;

    return;
}

static c_int _count_diagonal_entries(csc* P){

  c_int j;
  c_int count = 0;

  for(j = 0; j < P->n; j++){
    //look for nonempty columns with final element
    //on the diagonal.  Assumes triu format.
    if((P->p[j+1] != P->p[j]) && (P->i[P->p[j+1]-1] == j) ){
      count++;
    }
  }
  return count;

}


static void _kkt_assemble_csr(
    csc* K,
    c_int* PtoKKT,
    c_int* AtoKKT,
    c_int* rhotoKKT,
    csc* P,
    csc* A)
{

    //NB:  assembling a TRIU KKT in CSR format,
    //which is the same as TRIL KKT in CSC.
    c_int j;
    c_int m = A->m;
    c_int n = P->n;

    //use K.p to hold nnz entries in each
    //column of the KKT matrix
    for(int j=0; j <= (m+n); j++){K->p[j] = 0;}
    _kkt_colcount_missing_diag(K,P,0);
    _kkt_colcount_block(K,P,0,1);
    _kkt_colcount_block(K,A,0,0);
    _kkt_colcount_diag(K,n,m);

    //cumsum total entries to convert to K.p
    _kkt_colcount_to_colptr(K);

    //fill in value for P, top left (transposed/rowwise)
    _kkt_fill_missing_diag_zeros(K,P,0);  //before adding P, since tril form
    _kkt_fill_block(K,P,PtoKKT,0,0,1);

    //fill in value for A, lower left (columnwise)
    _kkt_fill_block(K,A,AtoKKT,n,0,0);

    //fill in lower right with diagonal of structural zeros
    _kkt_fill_diag_zeros(K,rhotoKKT,n,m);

    //backshift the colptrs to recover K.p again
    _kkt_backshift_colptrs(K);

    return;
}

static void _kkt_assemble_csc(
    csc* K,
    c_int* PtoKKT,
    c_int* AtoKKT,
    c_int* rhotoKKT,
    csc* P,
    csc* A)
{

    c_int j;
    c_int m = A->m;
    c_int n = P->n;

    //use K.p to hold nnz entries in each
    //column of the KKT matrix
    for(int j=0; j <= (m+n); j++){K->p[j] = 0;}
    _kkt_colcount_block(K,P,0,0);
    _kkt_colcount_missing_diag(K,P,0);
    _kkt_colcount_block(K,A,n,1);
    _kkt_colcount_diag(K,n,m);

    //cumsum total entries to convert to K.p
    _kkt_colcount_to_colptr(K);

    //fill in value for P, top left (columnwise)
    _kkt_fill_block(K,P,PtoKKT,0,0,0);
    _kkt_fill_missing_diag_zeros(K,P,0);  //after adding P, since triu form

    //fill in value for A, lower left (transposed/rowwise)
    _kkt_fill_block(K,A,AtoKKT,0,n,1);

    //fill in lower right with diagonal of structural zeros
    _kkt_fill_diag_zeros(K,rhotoKKT,n,m);

    //backshift the colptrs to recover K.p again
    _kkt_backshift_colptrs(K);

    return;
}


csc* form_KKT(csc*        P,
              csc*        A,
              c_int       format,
              c_float     param1,
              c_float*    param2,
              c_float     param2_sc,
              c_int*      PtoKKT,
              c_int*      AtoKKT,
              c_int*      rhotoKKT) {

  c_int   m,n;            //number of variables, constraints
  c_int  nKKT, nnzKKT;    // Size, number of nonzeros in KKT
  c_int  ndiagP;          // entries on diagonal of P
  csc*   KKT;             // KKT matrix in CSC (or CSR) format
  c_int  ptr, i, j;       // Counters for elements (i,j) and index pointer
  c_int  zKKT = 0;        // Counter for total number of elements in P and in
                          // KKT
  c_int* KKT_TtoC;        // Pointer to vector mapping from KKT in triplet form
                          // to CSC

  // Get matrix dimensions
  m   = A->m;
  n   = P->n;
  nKKT = m + n;

  //count elements on diag(P)
  ndiagP = _count_diagonal_entries(P);

  // Get maximum number of nonzero elements (only upper triangular part)
  nnzKKT = P->p[n] +   // Number of elements in P
           n -         // Number of elements in param1 * I
           ndiagP +    //remove double count on the diagonal
           A->p[n] +   // Number of nonzeros in A
           m;          // Number of elements in - diag(param2)

  // Preallocate KKT matrix in csc format
  KKT = csc_spalloc(nKKT, nKKT, nnzKKT, 1, 0);
  if (!KKT) return OSQP_NULL;  // Failed to preallocate matrix

  if(format == 0){  //KKT should be built in CSC format
    _kkt_assemble_csc(KKT,PtoKKT,AtoKKT,rhotoKKT,P,A);
  }
  else {          //KKT should be built in CSR format
    _kkt_assemble_csr(KKT,PtoKKT,AtoKKT,rhotoKKT,P,A);
  }
  //apply positive shifts to upper LH diagonal
  _kkt_shifts_param1(KKT,param1,n,format);

  //apply negative shifts to lower RH diagonal
  //NB: rhtoKKT is not needed to do this
  _kkt_shifts_param2(KKT,param2,param2_sc,n,m,format);

  return KKT;
}

#endif /* ifndef EMBEDDED */


#if EMBEDDED != 1

void update_KKT_P(csc*         KKT,
                  csc*         P,
                  const c_int* Px_new_idx,
                  c_int        P_new_n,
                  c_int*       PtoKKT,
                  c_float      param1,
                  c_int        format)
{
  c_int j, Pidx, Kidx, row, offset, doall;

  if(P_new_n <= 0){return;}

  //if Px_new_idx is null, we assume that all
  //elements are to be replaced (and that P_new_n = nnz(P))
  doall  = Px_new_idx == OSQP_NULL ? 1 : 0;
  offset = format == 0 ? 1 : 0;

  for (j = 0; j < P_new_n; j++) {
    Pidx = doall ? j : Px_new_idx[j];
    Kidx = PtoKKT[Pidx];
    KKT->x[Kidx] = P->x[Pidx];

    //is the corresonding column nonempty with
    //the current element on the diagonal (i.e. row==col)?
    row  = P->i[Pidx];
    if((P->p[row] < P->p[row+1]) && ((P->p[row+offset] - offset) == Pidx)){
      KKT->x[Kidx] += param1;
    }
  }
  return;
}

void update_KKT_A(csc*         KKT,
                  csc*         A,
                  const c_int* Ax_new_idx,
                  c_int        A_new_n,
                  c_int*       AtoKKT)
{

  c_int j, nnzA, Aidx, Kidx, doall;

  if(A_new_n <= 0){return;}

  //if Ax_new_idx is null, we assume that all
  //elements are to be replaced (and that A_new_n = nnz(A))
  doall  = Ax_new_idx == OSQP_NULL ? 1 : 0;

  // Update elements of KKT using A
  nnzA = A->p[A->n];
  for (j = 0; j < A_new_n; j++) {
    Aidx = doall ? j : Ax_new_idx[j];
    Kidx = AtoKKT[Aidx];
    KKT->x[Kidx] = A->x[Aidx];
  }
  return;
}


void update_KKT_param2(csc     *KKT,
                       c_float *param2,
                       c_float  param2_sc,
                       c_int   *param2toKKT,
                       c_int    m) {
  c_int i; // Iterations

  // Update elements of KKT using param2
  if (param2) {
    for (i = 0; i < m; i++) {
      KKT->x[param2toKKT[i]] = -param2[i];
    }
  }
  else {
    for (i = 0; i < m; i++) {
      KKT->x[param2toKKT[i]] = -param2_sc;
    }
  }
}

#endif // EMBEDDED != 1
