#!/bin/bash
OSQP=../../
gcc -c -g -Wimplicit-function-declaration osqp_f2c.c -I$OSQP/include
gfortran -c -g osqpf_types.F90
gfortran -c -g osqpf.F90
gfortran -c -g test_osqpf.F90
gfortran -o test_osqpf -I./ test_osqpf.o osqpf_types.o osqpf.o osqp_f2c.o -losqpstatic -L$OSQP/out -ldl -lm
./test_osqpf
rm *.o *.mod
rm test_osqpf
