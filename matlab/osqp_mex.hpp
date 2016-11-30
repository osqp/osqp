#ifndef __OSQP_MEX_HPP__
#define __OSQP_MEX_HPP__
#include "mex.h"
#include <stdint.h>
#include <string>
#include <cstring>
#include <typeinfo>

#define OSQP_MEX_SIGNATURE 0x271C1A7A
template<class base> class osqp_mex_handle
{
public:
    osqp_mex_handle(base *ptr) : ptr_m(ptr), name_m(typeid(base).name()) { signature_m = OSQP_MEX_SIGNATURE; }
    ~osqp_mex_handle() { signature_m = 0; delete ptr_m; }
    bool isValid() { return ((signature_m == OSQP_MEX_SIGNATURE) && !strcmp(name_m.c_str(), typeid(base).name())); }
    base *ptr() { return ptr_m; }

private:
    uint32_t signature_m;
    std::string name_m;
    base *ptr_m;
};

template<class base> inline mxArray *convertPtr2Mat(base *ptr)
{
    mexLock();
    mxArray *out = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    *((uint64_t *)mxGetData(out)) = reinterpret_cast<uint64_t>(new osqp_mex_handle<base>(ptr));
    return out;
}

template<class base> inline osqp_mex_handle<base> *convertMat2HandlePtr(const mxArray *in)
{
    if (mxGetNumberOfElements(in) != 1 || mxGetClassID(in) != mxUINT64_CLASS || mxIsComplex(in))
        mexErrMsgTxt("Input must be a real uint64 scalar.");
    osqp_mex_handle<base> *ptr = reinterpret_cast<osqp_mex_handle<base> *>(*((uint64_t *)mxGetData(in)));
    if (!ptr->isValid())
        mexErrMsgTxt("Handle not valid.");
    return ptr;
}

template<class base> inline base *convertMat2Ptr(const mxArray *in)
{
    return convertMat2HandlePtr<base>(in)->ptr();
}

template<class base> inline void destroyObject(const mxArray *in)
{
    delete convertMat2HandlePtr<base>(in);
    mexUnlock();
}

#endif // __OSQP_MEX_HPP__
