// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2022-2023 california institute of technology
// all rights reserved

/**
 * detail.cuh
 * common shared routines
 * norm, sum, mean of a vector
 **/


// code guard
#ifndef cuda_ode_detail_cuh
#define cuda_ode_detail_cuh

#include "external.h"

// check atomicAdd double is defined
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
static inline __device__ double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

// The default implementation for atomic maximum
template <typename T>
inline __device__ void AtomicMax(T * const address, const T value)
{
	atomicMax(address, value);
}


template <>
inline __device__ void AtomicMax(float * const address, const float value)
{
	if (* address >= value)
	{
		return;
	}

	int * const address_as_i = (int *)address;
	int old = * address_as_i, assumed;

	do
	{
		assumed = old;
		if (__int_as_float(assumed) >= value)
		{
			break;
		}

		old = atomicCAS(address_as_i, assumed, __float_as_int(value));
	} while (assumed != old);
}

template <>
inline __device__ void AtomicMax(double * const address, const double value)
{
	if (* address >= value)
	{
		return;
	}

	unsigned long long * const address_as_i = (unsigned long long *)address;
    unsigned long long old = * address_as_i, assumed;

	do
	{
        assumed = old;
		if (__longlong_as_double(assumed) >= value)
		{
			break;
		}

        old = atomicCAS(address_as_i, assumed, __double_as_longlong(value));
    } while (assumed != old);
}

namespace cuda::detail {


// sum reduction within a thread block for values returned by func(args...)
template<class T, class FuncType, class... Args>
__device__  auto sum_block(
    const cg::thread_block & cta,
    const int N,
    FuncType func,
    Args... args) -> T
{
    // each thread tid sum errors over elements of tid, tid+block_size, tid+2*block_size ...
    T thread_sum = static_cast<T>(0);
    for(int i=cta.thread_rank(); i<N; i+=cta.size())
    {
        auto sk = func(i, args...);
        thread_sum += sk;
    }

    // sum over each warp (32-threads tile)
    auto tile = cg::tiled_partition<32>(cta);
    auto tile_sum = cg::reduce(tile, thread_sum, cg::plus<T>());

    // define/init the block sum
	__shared__ T sum;
    if(cta.thread_rank()==0)
    {
		sum = static_cast<T>(0);
	}
	cta.sync();

	// sum over all tiles
    if(tile.thread_rank()==0)
    {
        atomicAdd(&sum, tile_sum);
    }
    cta.sync();

    if(cta.thread_rank()==0)
        return sum;
}

// max reduction within a thread block for values returned by func(args...)
// assume all elements are positive
template<class T, class FuncType, class... Args>
__device__  auto max_block(
    const cg::thread_block & cta,
    const int N,
    FuncType func,
    Args... args) -> T
{
    // each thread tid sum errors over elements of tid, tid+block_size, tid+2*block_size ...
    T thread_max = static_cast<T>(0);
    for(int i=cta.thread_rank(); i<N; i+=cta.size())
    {
        auto sk = func(i, args...);
        thread_max = max(thread_max, sk);
    }

    // max over each warp (32-threads tile)
    auto tile = cg::tiled_partition<32>(cta);
    auto tile_max = cg::reduce(tile, thread_max, cg::greater<T>());

    // define/init the block max
	__shared__ T max_val;
    if(cta.thread_rank()==0)
    {
		max_val = static_cast<T>(0);
	}
	cta.sync();

	// max over all tiles
    if(tile.thread_rank()==0)
    {
        AtomicMax<T>(&max_val, tile_max);
    }
    cta.sync();

    if(cta.thread_rank()==0)
        return max_val;
}

template<class T>
__device__  auto sum(
    const cg::thread_block & cta,
    const int N,
    const T* data) -> T
{
    auto func = [=] (const int i) -> T
        {
            return data[i];
        };
    auto sum = sum_block<T, decltype(func)>(cta, N, func);
    return sum;
}

template<class T>
__device__  auto mean(
    const cg::thread_block & cta,
    const int N,
    const T* data) -> T
{
    return sum(cta, N, data)/N;
}


template<class T>
__device__  auto norm(
    const cg::thread_block & cta,
    const int N,
    const T* data) -> T
{
    auto func = [=] (const int i) -> T {
        auto val = data[i];
        return val*val;
    };
    auto sum_sq = sum_block<T, decltype(func)>(cta, N, func);
    return sqrt(sum_sq);
}

template<class T>
__device__  auto norm_mean(
    const cg::thread_block & cta,
    const int N,
    const T* data) -> T
{
    return norm(cta, N, data)/sqrt(N);
}

template<class T>
__device__  void vector_copy(const cg::thread_block & cta, T* dst, const T* src, const int N)
{
    for(int i=cta.thread_rank(); i<N; i+=cta.size())
        dst[i] = src[i];
}


// block process template
// use a thread block to process func(i, args...), i \in [0,N) is the element index
// example: to set y[i] = a* x[i] + b
//     auto axpy = [=] (const int i) {y[i] += a*x[i]+b;};
//     block_process<T, decltype(axpy)>(N, axpy);
template<class T, class ProcType, class... Args>
__device__ auto block_process(
    const int N,
    const ProcType & func,
    Args... args)
{
    auto cta = cg::this_thread_block();
    for (auto tid = cta.thread_rank(); tid<N; tid+=cta.size())
        func(tid, args...);
}

} // end of namespace

#endif //__cuda_ode_detail_cuh__
// end of file
