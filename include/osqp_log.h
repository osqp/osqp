#ifndef OSQP_LOG_H
#define OSQP_LOG_H

#ifdef __cplusplus
extern "C" {
#endif /* ifdef __cplusplus */

/* Maximum length of the log file name */
#define OSQP_LOG_NAME_SIZE 255

/**
 * Assigns the name of the log file and opens the file for writing.
 *
 * @param log_name the path name of the log file to open.
 *
 * @return 0 if successful, non-zero otherwise.
 */
int osqp_open_log(const char* log_name);

/**
 * Closes the log file if it is open; otherwise, a no-op.
 */
void osqp_close_log();

/**
 * Writes a message to the console and to the log file (if a log
 * file has been opened).
 *
 * @param format the printf format.
 * @param ...    additional arguments to printf.
 *
 * @return the number of characters written, if successful, or -1
 * otherwise.
 */
int osqp_log(const char* format, ...);

#ifdef __cplusplus
}
#endif /* ifdef __cplusplus */

#endif /* ifndef OSQP_LOG_H */
