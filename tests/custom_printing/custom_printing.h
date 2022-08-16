#ifndef CUSTOM_PRINTING_H
#define CUSTOM_PRINTING_H

# ifdef __cplusplus
extern "C" {
# endif

/* We will provide our own error function, so remove the default one */
#undef c_eprint

/* Must have the same signature as printf for both macros */
#define c_print       my_print
#define c_eprint(...) my_eprint( __FUNCTION__, __VA_ARGS__ )

/* Implemented inside custom_printing.c */
void my_print(const char* format, ...);
void my_eprint(const char* funcName, const char* format, ...);

# ifdef __cplusplus
}
# endif

#endif /* ifndef  CUSTOM_PRINTING_H */
