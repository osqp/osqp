#include "custom_printing.h"
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>

/* Make a global counter with the number of lines/errors printed */
long int line_counter = 0;
long int error_counter = 0;

void my_print(const char* format, ...) {
  line_counter = line_counter + 1;

  /* Add a number to the begining of each line counting how many occurred*/
  va_list print_args;
  va_start( print_args, format);

  printf( "Line %ld: ", line_counter );
  printf( format, print_args );

  va_end(print_args);
}


void my_eprint(const char* funcName, const char* format, ...) {
  error_counter = error_counter + 1;

  /* Add a number to the begining of each line counting how many errors occurred*/
  va_list print_args;
  va_start( print_args, format);

  printf( "Error %ld: ", error_counter );
  printf( "ERROR in %s: ", funcName );
  printf( format, print_args );
  printf( "\n" );

  va_end(print_args);
}
