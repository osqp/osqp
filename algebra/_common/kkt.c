#include "kkt.h"


//add an offset to every term in the upper nxn block.
//assumes triu CSC or CSR format, with fully populated diagonal.
//format = 0 / CSC:  diagonal terms are last in every column.
//format = 1 / CSR:  diagonal terms are first in every row.
static void _kkt_shifts_param1(OSQPCscMatrix* KKT,
                               OSQPFloat      param1,
                               OSQPInt        n,
                               OSQPInt        format) {
  OSQPInt i;
  OSQPInt offset = format == 0 ? 1 : 0;
  for(i = 0; i < n; i++){ KKT->x[KKT->p[i+offset]-offset] += param1;}
  return;
}

//*subtract* an offset to every term in the lower mxm block.
//assumes triu CSC P/A formats, with fully populated diagonal.
//KKT format = 0 / CSC:  diagonal terms are last in every column.
//KKT format = 1 / CSR:  diagonal terms are first in every row.
static void _kkt_shifts_param2(OSQPCscMatrix* KKT,
                               OSQPFloat*     param2,
                               OSQPFloat      param2_sc,
                               OSQPInt        startcol,
                               OSQPInt        blockwidth,
                               OSQPInt        format) {

  OSQPInt i;
  OSQPInt offset = format == 0 ? 1 : 0;

  if(param2){
    for(i = 0; i < blockwidth; i++){ KKT->x[KKT->p[i + startcol + offset]-offset] -= param2[i];}
  }else{
    for(i = 0; i < blockwidth; i++){ KKT->x[KKT->p[i + startcol + offset]-offset] -= param2_sc;}
  }
}

#ifndef OSQP_EMBEDDED_MODE

//the remainder of the private functions here are for KKT
//assembly ONLY, so don't included them when OSQP_EMBEDDED_MODE

//increment the K colptr by the number of nonzeros
//in a square diagonal matrix placed on the diagonal.
//Used to increment, e.g. the lower RHS block diagonal
static void _kkt_colcount_diag(OSQPCscMatrix* K,
                               OSQPInt        initcol,
                               OSQPInt        blockcols) {

    OSQPInt j;
    for(j = initcol; j < (initcol + blockcols); j++){
        K->p[j]++;
    }
    return;
}

//same as _kkt_colcount_diag, but counts places
//where the input matrix M has a missing
//diagonal entry.  M must be square and TRIU
static void _kkt_colcount_missing_diag(OSQPCscMatrix* K,
                                       OSQPCscMatrix* M,
                                       OSQPInt        initcol) {

    OSQPInt j;
    for (j = 0; j < M->n; j++){
        //if empty column or last entry not on diagonal..
        if((M->p[j] == M->p[j+1]) || (M->i[M->p[j+1]-1] != j)) {
            K->p[j + initcol]++;
        }
    }
    return;
}

//increment K colptr by the number of nonzeros in M
static void _kkt_colcount_block(OSQPCscMatrix* K,
                                OSQPCscMatrix* M,
                                OSQPInt        initcol,
                                OSQPInt        istranspose) {

    OSQPInt nnzM, j;

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
static void _kkt_fill_block(OSQPCscMatrix* K,
                            OSQPCscMatrix* M,
                            OSQPInt*       MtoKKT,
                            OSQPInt        initrow,
                            OSQPInt        initcol,
                            OSQPInt        istranspose) {
    OSQPInt ii, jj, row, col, dest;

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
static void _kkt_fill_diag_zeros(OSQPCscMatrix* K,
                                 OSQPInt*       rhotoKKT,
                                 OSQPInt        offset,
                                 OSQPInt        blockdim) {

    OSQPInt j, dest, col;
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
static void _kkt_fill_missing_diag_zeros(OSQPCscMatrix* K,
                                         OSQPCscMatrix* M,
                                         OSQPInt        offset) {

    OSQPInt j, dest;
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

static void _kkt_colcount_to_colptr(OSQPCscMatrix* K) {

    OSQPInt j, count;
    OSQPInt currentptr = 0;

    for(j = 0; j <= K->n; j++){
       count        = K->p[j];
       K->p[j]      = currentptr;
       currentptr  += count;
    }
    return;
}

static void _kkt_backshift_colptrs(OSQPCscMatrix* K) {

    int j;
    for(j = K->n; j > 0; j--){
        K->p[j] = K->p[j-1];
    }
    K->p[0] = 0;

    return;
}

static OSQPInt _count_diagonal_entries(OSQPCscMatrix* P) {

  OSQPInt j;
  OSQPInt count = 0;

  for(j = 0; j < P->n; j++){
    //look for nonempty columns with final element
    //on the diagonal.  Assumes triu format.
    if((P->p[j+1] != P->p[j]) && (P->i[P->p[j+1]-1] == j) ){
      count++;
    }
  }
  return count;

}


static void _kkt_assemble_csr(OSQPCscMatrix* K,
                              OSQPInt*       PtoKKT,
                              OSQPInt*       AtoKKT,
                              OSQPInt*       rhotoKKT,
                              OSQPCscMatrix* P,
                              OSQPCscMatrix* A) {

    //NB:  assembling a TRIU KKT in CSR format,
    //which is the same as TRIL KKT in CSC.
    OSQPInt j;
    OSQPInt m = A->m;
    OSQPInt n = P->n;

    //use K.p to hold nnz entries in each
    //column of the KKT matrix
    for(j=0; j <= (m+n); j++){K->p[j] = 0;}
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

static void _kkt_assemble_csc(OSQPCscMatrix* K,
                              OSQPInt*       PtoKKT,
                              OSQPInt*       AtoKKT,
                              OSQPInt*       rhotoKKT,
                              OSQPCscMatrix* P,
                              OSQPCscMatrix* A) {

    OSQPInt j;
    OSQPInt m = A->m;
    OSQPInt n = P->n;

    //use K.p to hold nnz entries in each
    //column of the KKT matrix
    for(j=0; j <= (m+n); j++){K->p[j] = 0;}
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


OSQPCscMatrix* form_KKT(OSQPCscMatrix* P,
                        OSQPCscMatrix* A,
                        OSQPInt        format,
                        OSQPFloat      param1,
                        OSQPFloat*     param2,
                        OSQPFloat      param2_sc,
                        OSQPInt*       PtoKKT,
                        OSQPInt*       AtoKKT,
                        OSQPInt*       rhotoKKT) {

  OSQPInt   m,n;            //number of variables, constraints
  OSQPInt  nKKT, nnzKKT;    // Size, number of nonzeros in KKT
  OSQPInt  ndiagP;          // entries on diagonal of P

  OSQPCscMatrix* KKT;     // KKT matrix in CSC (or CSR) format

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

#endif /* ifndef OSQP_EMBEDDED_MODE */


#if OSQP_EMBEDDED_MODE != 1

void update_KKT_P(OSQPCscMatrix* KKT,
                  OSQPCscMatrix* P,
                  const OSQPInt* Px_new_idx,
                  OSQPInt        P_new_n,
                  OSQPInt*       PtoKKT,
                  OSQPFloat      param1,
                  OSQPInt        format) {
  OSQPInt j, Pidx, Kidx, row, offset, doall;

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

void update_KKT_A(OSQPCscMatrix* KKT,
                  OSQPCscMatrix* A,
                  const OSQPInt* Ax_new_idx,
                  OSQPInt        A_new_n,
                  OSQPInt*       AtoKKT) {

  OSQPInt j, Aidx, Kidx, doall;

  if(A_new_n <= 0){return;}

  //if Ax_new_idx is null, we assume that all
  //elements are to be replaced (and that A_new_n = nnz(A))
  doall  = Ax_new_idx == OSQP_NULL ? 1 : 0;

  // Update elements of KKT using A
  for (j = 0; j < A_new_n; j++) {
    Aidx = doall ? j : Ax_new_idx[j];
    Kidx = AtoKKT[Aidx];
    KKT->x[Kidx] = A->x[Aidx];
  }
  return;
}


void update_KKT_param2(OSQPCscMatrix* KKT,
                       OSQPFloat*     param2,
                       OSQPFloat      param2_sc,
                       OSQPInt*       param2toKKT,
                       OSQPInt        m) {
  OSQPInt i; // Iterations

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

#endif // OSQP_EMBEDDED_MODE != 1
