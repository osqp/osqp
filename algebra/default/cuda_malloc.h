#ifndef CUDA_MALLOC_H
# define CUDA_MALLOC_H


#define c_cudaMalloc cudaMalloc

template<typename T>
inline cudaError_t c_cudaFree(T** devPtr) {
	cudaError_t cuda_error = cudaFree(*devPtr);
	*devPtr = NULL;
	return cuda_error;
}

template<typename T>
inline cudaError_t  c_cudaCalloc(T** devPtr, size_t size) {
	cudaError_t cudaCalloc_er = cudaMalloc(devPtr, size);
	if (cudaCalloc_er == cudaSuccess) {
		return cudaMemset(*devPtr,0,size);
	}
	else {
		return cudaCalloc_er;
	}
}


#endif /* ifndef CUDA_MALLOC_H */
