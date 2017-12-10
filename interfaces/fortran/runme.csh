#!/bin/csh -f
set OSQP='../../'
gcc -c -g osqp_f2c.c -I$OSQP/include
echo $status
if ($status == 0) gfortran -c -g osqpf_types.F90
if ($status == 0) gfortran -c -g osqpf.F90
if ($status == 0) gfortran -o test_osqpf -I./ test_osqpf.F90 osqpf_types.o osqpf.o osqp_f2c.o -losqpstatic -L$OSQP/out -ldl -lm
if ($status == 0) then
  ./test_osqpf
  rm test_osqpf
endif
rm -f *.o *.mod
