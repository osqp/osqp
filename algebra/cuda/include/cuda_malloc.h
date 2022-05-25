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

#ifndef CUDA_MALLOC_H
# define CUDA_MALLOC_H


void cuda_malloc(void** devPtr, size_t size);

void cuda_malloc_host(void** devPtr, size_t size);

void cuda_calloc(void** devPtr, size_t size);

void cuda_free(void** devPtr);

void cuda_free_host(void** devPtr);


#endif /* ifndef CUDA_MALLOC_H */
