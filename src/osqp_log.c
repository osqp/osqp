#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <sys/errno.h>
#include "osqp_log.h"

static FILE* osqp_log_fp = NULL;
static char  osqp_log_name[OSQP_LOG_NAME_SIZE + 1];

int osqp_open_log(const char* log_name) {
  if (strlen(log_name) > OSQP_LOG_NAME_SIZE) {
    fprintf(stderr, "WARNING: Log file name exceeds the allowed size.\n");
    return -1;
  }

  osqp_close_log();
  strcpy(osqp_log_name, log_name);
  osqp_log_fp = fopen(osqp_log_name, "w");

  if (osqp_log_fp)
    return 0;
  else
    return errno;
}

void osqp_close_log() {
  if (osqp_log_fp) {
    fclose(osqp_log_fp);
    osqp_log_fp = NULL;
  }
}

int osqp_log(const char* format, ...) {
  va_list args;

  va_start(args, format);
  int result = vprintf(format, args);
  va_end(args);

  if (osqp_log_fp) {
    va_start(args, format);
    result = vfprintf(osqp_log_fp, format, args);
    va_end(args);
  }

  return result;
}
