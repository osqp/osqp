/**
 *  Copyright (c) 2019-2021 ETH Zurich, Automatic Control Lab,
 *  Michel Schubiger, Goran Banjac.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#ifndef CUDA_MEMORY_H
#define CUDA_MEMORY_H

#include <cuda_runtime.h>

void cuda_malloc(void** devPtr, size_t size);

void cuda_malloc_host(void** devPtr, size_t size);

void cuda_calloc(void** devPtr, size_t size);

void cuda_free(void** devPtr);

void cuda_free_host(void** devPtr);

/**
 * Test if a pointer points to a region of memory on the device.
 *
 * @return true if the pointer is to memory on the device, false otherwise
 */
bool cuda_isdeviceptr(const void* ptr);

/**
 * Copy data from either host or device memory to device memory.
 *
 * This will automatically test the location of src to see if it is host
 * or device memory, and perform the appropriate copy.
 *
 * @param dst   Destination pointer on the device
 * @param src   Source pointer on either the host or device
 * @param count Number of bytes to copy
 * @return      Error code
 */
cudaError_t cuda_memcpy_hd2d(void* dst, const void* src, size_t count);

#endif /* ifndef CUDA_MEMORY_H */
