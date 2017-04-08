/* ========================================================================== */
/* === ldl.h:  include file for the LDL package ============================= */
/* ========================================================================== */

/* Copyright (c) Timothy A Davis, http://www.suitesparse.com.
 * All Rights Reserved.  See LDL/Doc/License.txt for the License.
 */


#include "types.h"

#define LDL_int c_int

#ifdef DLONG

#ifndef EMBEDDED
#include "SuiteSparse_config.h"
#define LDL_symbolic ldl_l_symbolic
#else
#define SuiteSparse_long c_int
#endif

#if EMBEDDED != 1
#define LDL_numeric ldl_l_numeric
#endif

#define LDL_lsolve ldl_l_lsolve
#define LDL_dinvsolve ldl_l_dinvsolve
#define LDL_ltsolve ldl_l_ltsolve
#define LDL_perm ldl_l_perm
#define LDL_permt ldl_l_permt
// #define LDL_dsolve ldl_l_dsolve
// #define LDL_valid_perm ldl_l_valid_perm
// #define LDL_valid_matrix ldl_l_valid_matrix

#else  // DLONG

#ifndef EMBEDDED
#include "SuiteSparse_config.h"
#define LDL_symbolic ldl_symbolic
#else
#define SuiteSparse_long c_int
#endif


#if EMBEDDED != 1
#define LDL_numeric ldl_numeric
#endif

#define LDL_lsolve ldl_lsolve
// #define LDL_dsolve ldl_dsolve
#define LDL_dinvsolve ldl_dinvsolve
#define LDL_ltsolve ldl_ltsolve
#define LDL_perm ldl_perm
#define LDL_permt ldl_permt
// #define LDL_valid_perm ldl_valid_perm
// #define LDL_valid_matrix ldl_valid_matrix

#endif  // DLONG

/* ========================================================================== */
/* === int version ========================================================== */
/* ========================================================================== */

#ifndef EMBEDDED
void ldl_symbolic (int n, int Ap [ ], int Ai [ ], int Lp [ ],
    int Parent [ ], int Lnz [ ], int Flag [ ], int P [ ], int Pinv [ ]) ;
#endif

#if EMBEDDED != 1
int ldl_numeric (int n, int Ap [ ], int Ai [ ], c_float Ax [ ],
    int Lp [ ], int Parent [ ], int Lnz [ ], int Li [ ], c_float Lx [ ],
    c_float D [ ], c_float Y [ ], int Pattern [ ], int Flag [ ],
    int P [ ], int Pinv [ ]) ;
#endif

void ldl_lsolve (int n, c_float X [ ], int Lp [ ], int Li [ ],
    c_float Lx [ ]) ;

// void ldl_dsolve (int n, c_float X [ ], c_float D [ ]) ;

void ldl_dinvsolve (int n, c_float X [ ], c_float Dinv [ ]) ;

void ldl_ltsolve (int n, c_float X [ ], int Lp [ ], int Li [ ],
    c_float Lx [ ]) ;

void ldl_perm  (int n, c_float X [ ], c_float B [ ], int P [ ]) ;
void ldl_permt (int n, c_float X [ ], c_float B [ ], int P [ ]) ;

int ldl_valid_perm (int n, int P [ ], int Flag [ ]) ;
int ldl_valid_matrix ( int n, int Ap [ ], int Ai [ ]) ;

/* ========================================================================== */
/* === long version ========================================================= */
/* ========================================================================== */

#ifndef EMBEDDED
void ldl_l_symbolic (SuiteSparse_long n, SuiteSparse_long Ap [ ],
    SuiteSparse_long Ai [ ], SuiteSparse_long Lp [ ],
    SuiteSparse_long Parent [ ], SuiteSparse_long Lnz [ ],
    SuiteSparse_long Flag [ ], SuiteSparse_long P [ ],
    SuiteSparse_long Pinv [ ]) ;
#endif

#if EMBEDDED != 1
SuiteSparse_long ldl_l_numeric (SuiteSparse_long n, SuiteSparse_long Ap [ ],
    SuiteSparse_long Ai [ ], c_float Ax [ ], SuiteSparse_long Lp [ ],
    SuiteSparse_long Parent [ ], SuiteSparse_long Lnz [ ],
    SuiteSparse_long Li [ ], c_float Lx [ ], c_float D [ ], c_float Y [ ],
    SuiteSparse_long Pattern [ ], SuiteSparse_long Flag [ ],
    SuiteSparse_long P [ ], SuiteSparse_long Pinv [ ]) ;
#endif

void ldl_l_lsolve (SuiteSparse_long n, c_float X [ ], SuiteSparse_long Lp [ ],
    SuiteSparse_long Li [ ], c_float Lx [ ]) ;

// void ldl_l_dsolve (SuiteSparse_long n, c_float X [ ], c_float D [ ]) ;
//

// x = Dinv*x
void ldl_l_dinvsolve(SuiteSparse_long n, c_float X [ ], c_float Dinv [ ]);

void ldl_l_ltsolve (SuiteSparse_long n, c_float X [ ], SuiteSparse_long Lp [ ],
    SuiteSparse_long Li [ ], c_float Lx [ ]) ;

void ldl_l_perm  (SuiteSparse_long n, c_float X [ ], c_float B [ ],
    SuiteSparse_long P [ ]) ;
void ldl_l_permt (SuiteSparse_long n, c_float X [ ], c_float B [ ],
    SuiteSparse_long P [ ]) ;

/*
SuiteSparse_long ldl_l_valid_perm (SuiteSparse_long n, SuiteSparse_long P [ ],
    SuiteSparse_long Flag [ ]) ;
SuiteSparse_long ldl_l_valid_matrix ( SuiteSparse_long n,
    SuiteSparse_long Ap [ ], SuiteSparse_long Ai [ ]) ;
 */

/* ========================================================================== */
/* === LDL version ========================================================== */
/* ========================================================================== */

#define LDL_DATE "May 4, 2016"
#define LDL_VERSION_CODE(main,sub) ((main) * 1000 + (sub))
#define LDL_MAIN_VERSION 2
#define LDL_SUB_VERSION 2
#define LDL_SUBSUB_VERSION 6
#define LDL_VERSION LDL_VERSION_CODE(LDL_MAIN_VERSION,LDL_SUB_VERSION)
