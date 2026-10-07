// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2026 california institute of technology
// all rights reserved

/**
 * stepper.cuh
 * Radau IIA (order 5) stepper for stiff systems, one system per thread block; follows
 * scipy.integrate.Radau: simplified Newton iterations on the collocation system, with the
 * Jacobian and the LU factorizations reused across steps while they keep converging
 *
 * an ode may declare {inert_size()}, the number of its leading components the derivatives do not
 * depend on (e.g., the slip in the seas models); the Newton systems are then solved on the
 * remaining components only, and the leading ones follow from them
 **/

// code guard
#ifndef cuda_ode_radau5_stepper_cuh
#define cuda_ode_radau5_stepper_cuh

#include <dopri5/external.h>
#include <dopri5/detail.cuh>
#include "linalg.cuh"

namespace cuda::ode::radau5 {

// the coefficients, from scipy.integrate.Radau
namespace coefficients {
    // the collocation nodes
    constexpr double C0 = 0.15505102572168222;
    constexpr double C1 = 0.6449489742783178;
    constexpr double C2 = 1.0;
    // the error estimate
    constexpr double E0 = -10.048809399827414;
    constexpr double E1 = 1.382142733160748;
    constexpr double E2 = -0.3333333333333333;
    // the eigenvalues of the inverse of the Butcher matrix
    constexpr double MU_REAL = 3.637834252744496;
    constexpr double MU_COMPLEX_RE = 2.6810828736277523;
    constexpr double MU_COMPLEX_IM = -3.050430199247411;
    // its eigenvectors, and their inverse
    constexpr double T00 = 0.09443876248897524, T01 = -0.1412552950209542, T02 = 0.03002919410514742;
    constexpr double T10 = 0.2502131229653333, T11 = 0.20412935229379994, T12 = -0.3829421127572619;
    constexpr double T20 = 1.0, T21 = 1.0, T22 = 0.0;
    constexpr double TI00 = 4.178718591551904, TI01 = 0.32768282076106237, TI02 = 0.5233764454994495;
    constexpr double TI10 = -4.178718591551904, TI11 = -0.32768282076106237, TI12 = 0.47662355450055044;
    constexpr double TI20 = 0.5028726349457868, TI21 = -2.571926949855605, TI22 = 0.5960392048282249;
    // the dense output
    constexpr double P00 = 10.048809399827414, P01 = -25.62959144707664, P02 = 15.580782047249224;
    constexpr double P10 = -1.382142733160748, P11 = 10.296258113743303, P12 = -8.914115380582556;
    constexpr double P20 = 0.3333333333333333, P21 = -2.6666666666666665, P22 = 3.3333333333333335;
    // the most Newton iterations in a step
    constexpr int NEWTON_MAXITER = 6;
}

// the number of leading components the derivatives of {ode} do not depend on
template <class ode_system_type>
__host__ __device__ int inert_size(const ode_system_type & ode)
{
    if constexpr (requires { ode.inert_size(); })
        return ode.inert_size();
    else
        return 0;
}

// the stepper state of one system
template <class T>
struct __ALIGNED__ Stepper {
    using complex_type = cuda::std::complex<T>;

    int patches;
    int units;
    int system_size; // N
    int lead; // the leading components the derivatives do not depend on
    int newton_size; // M = N - lead, the size of the Newton systems

    T* y0; // [N] initial value
    T* yn; // [N] final value at t0+h
    T* f0; // [N] f(t0, y0)
    T* Z; // [3, N] the stage increments
    T* W; // [3, N] the stage increments, transformed
    T* F; // [3, N] the stage derivatives
    T* dW; // [3, N] the Newton correction
    T* Q; // [3, N] the dense output coefficients of the last accepted step
    T* scale; // [N] error scale
    T* err; // [N] error estimate
    T* ys; // [N] a stage state
    T* fs; // [N] its derivative
    T* rr; // [N] the real Newton system
    complex_type* rc; // [N] the complex Newton system
    T* J; // [N, M] the Jacobian columns of the non inert components
    T* LUr; // [M, M] the factored real Newton matrix
    complex_type* LUc; // [M, M] the factored complex Newton matrix
    int* pivr; // [M] its pivots
    int* pivc; // [M]

    // tolerances, set by the controller
    T atol;
    T rtol;
    T newton_tol;

    // the step state
    bool f0_valid; // whether f0 = f(t0, y0)
    bool jac_valid; // whether J is usable
    bool jac_current; // whether J was computed at (t0, y0)
    bool lu_valid; // whether LUr, LUc are factored, for h = lu_h
    bool singular; // whether a factorization failed
    T lu_h;
    bool pred_valid; // whether Q predicts the next step
    T h_prev; // the step Q belongs to
    bool rejected; // whether the step was rejected before, set by the controller
    bool recompute_jac; // whether the Newton iterations asked for a new J, set by the controller
    bool newton_converged;
    int n_iter;
    T rate;
    T error_norm;

    // the work space of a system, in reals, complex numbers, and ints
    __host__ __device__ static size_t real_size(const int N, const int M) { return 23*(size_t)N + (size_t)N*M + (size_t)M*M; }
    __host__ __device__ static size_t complex_size(const int N, const int M) { return (size_t)N + (size_t)M*M; }
    __host__ __device__ static size_t int_size(const int N, const int M) { return 2*(size_t)M; }

    // initialize the pointers
    __device__ void init(const int pps, const int upp, const int lead_,
                         T* work, complex_type* cwork, int* iwork)
    {
        patches = pps;
        units = upp;
        system_size = patches*units;
        lead = lead_;
        newton_size = system_size - lead;
        auto N = system_size;
        auto M = newton_size;
        y0 = work; yn = y0 + N; f0 = yn + N;
        Z = f0 + N; W = Z + 3*N; F = W + 3*N; dW = F + 3*N; Q = dW + 3*N;
        scale = Q + 3*N; err = scale + N; ys = err + N; fs = ys + N; rr = fs + N;
        J = rr + N; LUr = J + (size_t)N*M;
        rc = cwork; LUc = rc + N;
        pivr = iwork; pivc = pivr + M;
        f0_valid = jac_valid = jac_current = lu_valid = singular = false;
        pred_valid = rejected = recompute_jac = newton_converged = false;
        lu_h = h_prev = rate = error_norm = static_cast<T>(0);
        n_iter = 0;
    }

    // forget everything that depends on (t0, y0), e.g., after an event changed y0
    __device__ void reset(const cg::thread_block & cta)
    {
        if (cta.thread_rank() == 0) {
            f0_valid = jac_valid = jac_current = lu_valid = singular = false;
            pred_valid = rejected = recompute_jac = false;
        }
        cta.sync();
    }

    template <class ode_system_type>
    __device__ void integrate(const cg::thread_block& cta, const int system_id, const T t0, const T h, ode_system_type& ode);

    template <class ode_system_type>
    __device__ void set_init_value(const cg::thread_block& cta,
            const T t0, const T* y0_in, const int system_id, ode_system_type& ode)
    {
        cuda::detail::vector_copy<T>(cta, y0, y0_in, system_size);
        cta.sync();
        reset(cta);
        set_f0_value(cta, t0, system_id, ode);
    }

    template <class ode_system_type>
    __device__ void set_f0_value(const cg::thread_block& cta,
            const T t0, const int system_id, ode_system_type& ode)
    {
        ode.dydt_block(cta, system_id, t0, y0, f0);
        cta.sync();
        if (cta.thread_rank() == 0)
            f0_valid = true;
        cta.sync();
    }

    // the rms norm of x/scale over n = k*N entries
    __device__ T rms(const cg::thread_block& cta, const T* x, const int n)
    {
        auto N = system_size;
        auto s = scale;
        auto f = [=] (const int i) -> T { auto v = x[i]/s[i % N]; return v*v; };
        auto sum = cuda::detail::sum_block<T, decltype(f)>(cta, n, f);
        return sqrt(sum/n);
    }

    template <class ode_system_type>
    __device__ void jacobian(const cg::thread_block& cta, const int system_id, const T t0, ode_system_type& ode);
    template <class ode_system_type>
    __device__ void jacobian_differences(const cg::thread_block& cta, const int system_id, const T t0,
                                         ode_system_type& ode);
    __device__ void factor(const cg::thread_block& cta, const T h);
    __device__ void solve_real(const cg::thread_block& cta, T* x, const T gamma);
    __device__ void solve_complex(const cg::thread_block& cta, complex_type* x, const complex_type gamma);
    template <class ode_system_type>
    __device__ bool newton(const cg::thread_block& cta, const int system_id, const T t0, const T h, ode_system_type& ode);
};


// the Jacobian columns of the non inert components at (t0, y0), from the ode if it knows them,
// else by forward differences
template <class T>
template <class ode_system_type>
__device__ void Stepper<T>::jacobian(const cg::thread_block& cta, const int system_id, const T t0,
                                     ode_system_type& ode)
{
    // an ode that knows its Jacobian provides it, from f0 = f(t0, y0), with fs as its work space
    if constexpr (requires { ode.jacobian_block(cta, system_id, t0, y0, f0, J, fs); })
        ode.jacobian_block(cta, system_id, t0, y0, f0, J, fs);
    else
        jacobian_differences(cta, system_id, t0, ode);

    if (cta.thread_rank() == 0) {
        jac_valid = true;
        jac_current = true;
        lu_valid = false;
    }
    cta.sync();
}

// the Jacobian columns of the non inert components at (t0, y0), by forward differences
template <class T>
template <class ode_system_type>
__device__ void Stepper<T>::jacobian_differences(const cg::thread_block& cta, const int system_id, const T t0,
                                                 ode_system_type& ode)
{
    auto N = system_size;
    auto M = newton_size;
    const auto eps = sqrt(cuda::std::numeric_limits<T>::epsilon());

    cuda::detail::vector_copy<T>(cta, ys, y0, N);
    cta.sync();
    for (auto c = 0; c < M; c++) {
        auto j = lead + c;
        auto yj = y0[j];
        // the perturbation, relative to the size of the component
        if (cta.thread_rank() == 0)
            ys[j] = yj + eps*max(abs(yj), static_cast<T>(1));
        cta.sync();
        // the perturbation, as it is represented
        auto d = ys[j] - yj;
        ode.dydt_block(cta, system_id, t0, ys, fs);
        cta.sync();
        for (auto i = static_cast<int>(cta.thread_rank()); i < N; i += cta.size())
            J[(size_t)i*M + c] = (fs[i] - f0[i])/d;
        if (cta.thread_rank() == 0)
            ys[j] = yj;
        cta.sync();
    }
}

// factor the Newton matrices gamma I - J22 for step h, with gamma = MU/h
template <class T>
__device__ void Stepper<T>::factor(const cg::thread_block& cta, const T h)
{
    using namespace coefficients;
    auto M = newton_size;
    auto gr = static_cast<T>(MU_REAL)/h;
    auto gc = complex_type(static_cast<T>(MU_COMPLEX_RE), static_cast<T>(MU_COMPLEX_IM))/h;
    for (auto idx = static_cast<int>(cta.thread_rank()); idx < M*M; idx += cta.size()) {
        auto i = idx / M;
        auto c = idx % M;
        auto Jic = J[(size_t)(lead + i)*M + c];
        LUr[idx] = (i == c ? gr : static_cast<T>(0)) - Jic;
        LUc[idx] = (i == c ? gc : complex_type(0)) - Jic;
    }
    cta.sync();
    auto sr = lu_factor<T, T>(cta, LUr, pivr, M);
    auto sc = lu_factor<T, complex_type>(cta, LUc, pivc, M);
    if (cta.thread_rank() == 0) {
        singular = sr || sc;
        lu_valid = true;
        lu_h = h;
    }
    cta.sync();
}

// solve (gamma I - J) x = b in place: the non inert components with the LU factors, the leading
// ones, whose diagonal block of J vanishes, from x_1 = (b_1 + J12 x_2) / gamma
template <class T>
__device__ void Stepper<T>::solve_real(const cg::thread_block& cta, T* x, const T gamma)
{
    auto M = newton_size;
    auto x2 = x + lead;
    lu_solve<T, T>(cta, LUr, pivr, x2, M);
    for (auto i = static_cast<int>(cta.thread_rank()); i < lead; i += cta.size()) {
        auto acc = x[i];
        for (auto c = 0; c < M; c++)
            acc += J[(size_t)i*M + c]*x2[c];
        x[i] = acc/gamma;
    }
    cta.sync();
}

template <class T>
__device__ void Stepper<T>::solve_complex(const cg::thread_block& cta, complex_type* x, const complex_type gamma)
{
    auto M = newton_size;
    auto x2 = x + lead;
    lu_solve<T, complex_type>(cta, LUc, pivc, x2, M);
    for (auto i = static_cast<int>(cta.thread_rank()); i < lead; i += cta.size()) {
        auto acc = x[i];
        for (auto c = 0; c < M; c++)
            acc += J[(size_t)i*M + c]*x2[c];
        x[i] = acc/gamma;
    }
    cta.sync();
}

// the simplified Newton iterations on the collocation system, starting from the guess in Z;
// returns whether they converged, with the stage increments in Z
template <class T>
template <class ode_system_type>
__device__ bool Stepper<T>::newton(const cg::thread_block& cta, const int system_id, const T t0, const T h,
                                   ode_system_type& ode)
{
    using namespace coefficients;
    auto N = system_size;
    auto gr = static_cast<T>(MU_REAL)/h;
    auto gc = complex_type(static_cast<T>(MU_COMPLEX_RE), static_cast<T>(MU_COMPLEX_IM))/h;
    const T c[3] = { static_cast<T>(C0), static_cast<T>(C1), static_cast<T>(C2) };

    // W = TI Z
    for (auto i = static_cast<int>(cta.thread_rank()); i < N; i += cta.size()) {
        auto z0 = Z[i], z1 = Z[N + i], z2 = Z[2*N + i];
        W[i] = static_cast<T>(TI00)*z0 + static_cast<T>(TI01)*z1 + static_cast<T>(TI02)*z2;
        W[N + i] = static_cast<T>(TI10)*z0 + static_cast<T>(TI11)*z1 + static_cast<T>(TI12)*z2;
        W[2*N + i] = static_cast<T>(TI20)*z0 + static_cast<T>(TI21)*z1 + static_cast<T>(TI22)*z2;
    }
    cta.sync();

    auto dW_norm_old = static_cast<T>(-1);
    auto rate_ = static_cast<T>(-1); // negative while unknown
    auto converged = false;
    auto k = 0;
    for (; k < NEWTON_MAXITER; k++) {
        // the stage derivatives
        for (auto s = 0; s < 3; s++) {
            for (auto i = static_cast<int>(cta.thread_rank()); i < N; i += cta.size())
                ys[i] = y0[i] + Z[s*N + i];
            cta.sync();
            ode.dydt_block(cta, system_id, t0 + c[s]*h, ys, F + s*N);
            cta.sync();
        }
        auto Fs = F;
        auto finite = [=] (const int i) -> T { return isfinite(Fs[i]) ? static_cast<T>(0) : static_cast<T>(1); };
        if (cuda::detail::sum_block<T, decltype(finite)>(cta, 3*N, finite) > static_cast<T>(0))
            break;

        // the right hand sides of the real and the complex Newton systems
        for (auto i = static_cast<int>(cta.thread_rank()); i < N; i += cta.size()) {
            auto F0 = F[i], F1 = F[N + i], F2 = F[2*N + i];
            rr[i] = static_cast<T>(TI00)*F0 + static_cast<T>(TI01)*F1 + static_cast<T>(TI02)*F2 - gr*W[i];
            rc[i] = complex_type(
                static_cast<T>(TI10)*F0 + static_cast<T>(TI11)*F1 + static_cast<T>(TI12)*F2,
                static_cast<T>(TI20)*F0 + static_cast<T>(TI21)*F1 + static_cast<T>(TI22)*F2)
                - gc*complex_type(W[N + i], W[2*N + i]);
        }
        cta.sync();
        solve_real(cta, rr, gr);
        solve_complex(cta, rc, gc);
        for (auto i = static_cast<int>(cta.thread_rank()); i < N; i += cta.size()) {
            dW[i] = rr[i];
            dW[N + i] = rc[i].real();
            dW[2*N + i] = rc[i].imag();
        }
        cta.sync();

        // the convergence of the iterations
        auto dW_norm = rms(cta, dW, 3*N);
        if (dW_norm_old >= static_cast<T>(0))
            rate_ = dW_norm/dW_norm_old;
        if (rate_ >= static_cast<T>(0) &&
            (rate_ >= static_cast<T>(1) ||
             pow(rate_, static_cast<T>(NEWTON_MAXITER - k))/(static_cast<T>(1) - rate_)*dW_norm > newton_tol))
            break;

        // W += dW, Z = T W
        for (auto i = static_cast<int>(cta.thread_rank()); i < N; i += cta.size()) {
            auto w0 = W[i] + dW[i], w1 = W[N + i] + dW[N + i], w2 = W[2*N + i] + dW[2*N + i];
            W[i] = w0; W[N + i] = w1; W[2*N + i] = w2;
            Z[i] = static_cast<T>(T00)*w0 + static_cast<T>(T01)*w1 + static_cast<T>(T02)*w2;
            Z[N + i] = static_cast<T>(T10)*w0 + static_cast<T>(T11)*w1 + static_cast<T>(T12)*w2;
            Z[2*N + i] = static_cast<T>(T20)*w0 + static_cast<T>(T21)*w1 + static_cast<T>(T22)*w2;
        }
        cta.sync();

        if (dW_norm == static_cast<T>(0) ||
            (rate_ >= static_cast<T>(0) && rate_/(static_cast<T>(1) - rate_)*dW_norm < newton_tol)) {
            converged = true;
            break;
        }
        dW_norm_old = dW_norm;
    }

    if (cta.thread_rank() == 0) {
        n_iter = k + 1;
        rate = rate_;
    }
    cta.sync();
    return converged;
}

/**
 * Radau step (t0, t0+h) with a cuda thread block: on success, yn and the error estimate;
 * whether it succeeded is left in {newton_converged}, the error in {error_norm}
 **/
template <class T>
template <class ode_system_type>
__device__ void Stepper<T>::integrate(const cg::thread_block& cta, const int system_id,
                                      const T t0, const T h, ode_system_type& ode)
{
    using namespace coefficients;
    auto N = system_size;
    auto tid = static_cast<int>(cta.thread_rank());
    auto block = static_cast<int>(cta.size());

    // f(t0, y0), and the Jacobian, if needed
    if (!f0_valid)
        set_f0_value(cta, t0, system_id, ode);
    if (!jac_valid)
        jacobian(cta, system_id, t0, ode);

    // the error scale of the Newton iterations
    for (auto i = tid; i < N; i += block)
        scale[i] = atol + abs(y0[i])*rtol;
    cta.sync();

    auto converged = false;
    while (true) {
        if (!lu_valid || lu_h != h)
            factor(cta, h);

        // the starting guess: the dense output of the previous step, or nothing
        if (pred_valid) {
            const T c[3] = { static_cast<T>(C0), static_cast<T>(C1), static_cast<T>(C2) };
            for (auto i = tid; i < N; i += block) {
                for (auto s = 0; s < 3; s++) {
                    auto x = static_cast<T>(1) + c[s]*h/h_prev;
                    Z[s*N + i] = Q[i]*(x - 1) + Q[N + i]*(x*x - 1) + Q[2*N + i]*(x*x*x - 1);
                }
            }
        }
        else {
            for (auto i = tid; i < 3*N; i += block)
                Z[i] = static_cast<T>(0);
        }
        cta.sync();

        converged = !singular && newton(cta, system_id, t0, h, ode);
        // done, or nothing left to try at this step
        if (converged || jac_current)
            break;
        // otherwise, retry with a fresh Jacobian
        jacobian(cta, system_id, t0, ode);
    }

    if (tid == 0)
        newton_converged = converged;
    cta.sync();
    if (!converged)
        return;

    // the new state, and the error estimate; ZE goes in F, the stage derivatives are spent
    auto ZE = F;
    for (auto i = tid; i < N; i += block) {
        auto z0 = Z[i], z1 = Z[N + i], z2 = Z[2*N + i];
        yn[i] = y0[i] + z2;
        ZE[i] = (static_cast<T>(E0)*z0 + static_cast<T>(E1)*z1 + static_cast<T>(E2)*z2)/h;
        err[i] = f0[i] + ZE[i];
        scale[i] = atol + max(abs(y0[i]), abs(yn[i]))*rtol;
    }
    cta.sync();
    auto gr = static_cast<T>(MU_REAL)/h;
    solve_real(cta, err, gr);
    auto en = rms(cta, err, N);

    // a step rejected before gets a better estimate
    if (rejected && en > static_cast<T>(1)) {
        for (auto i = tid; i < N; i += block)
            ys[i] = y0[i] + err[i];
        cta.sync();
        ode.dydt_block(cta, system_id, t0, ys, fs);
        cta.sync();
        for (auto i = tid; i < N; i += block)
            err[i] = fs[i] + ZE[i];
        cta.sync();
        solve_real(cta, err, gr);
        en = rms(cta, err, N);
    }

    if (tid == 0)
        error_norm = en;
    cta.sync();
}


template <class T>
__global__ void stepper_state_init_kernel(const int patches, const int units, const int lead,
    const int systems_batch, T* work, cuda::std::complex<T>* cwork, int* iwork, Stepper<T>* steppers)
{
    auto system = threadIdx.x + blockIdx.x*blockDim.x;
    if (system < systems_batch) {
        auto N = patches*units;
        auto M = N - lead;
        steppers[system].init(patches, units, lead,
            work + Stepper<T>::real_size(N, M)*system,
            cwork + Stepper<T>::complex_size(N, M)*system,
            iwork + Stepper<T>::int_size(N, M)*system);
    }
}

template <class T>
struct StepperHolder {
    int patches;
    int units;
    int lead;
    int systems_batch;
    Stepper<T> * steppers;
    T * work;
    cuda::std::complex<T> * cwork;
    int * iwork;

    // constructor, for the systems of {ode}
    template <class ode_system_type>
    StepperHolder(const ode_system_type & ode, const int sys)
        : patches(ode.patches), units(ode.units), lead(inert_size(ode)), systems_batch(sys)
    {
        auto N = patches*units;
        auto M = N - lead;
        cudaSafeCall(cudaMalloc(&work, Stepper<T>::real_size(N, M)*systems_batch*sizeof(T)));
        cudaSafeCall(cudaMalloc(&cwork, Stepper<T>::complex_size(N, M)*systems_batch*sizeof(cuda::std::complex<T>)));
        cudaSafeCall(cudaMalloc(&iwork, Stepper<T>::int_size(N, M)*systems_batch*sizeof(int)));
        cudaSafeCall(cudaMalloc(&steppers, systems_batch*sizeof(Stepper<T>)));
        int threads = 256;
        int blocks = (systems_batch-1+threads)/threads; // idivup
        stepper_state_init_kernel<T><<<blocks, threads>>>(patches, units, lead, systems_batch,
            work, cwork, iwork, steppers);
        cudaCheckError("radau5 stepper_state_init_kernel error");
    }

    ~StepperHolder() noexcept(false)
    {
        if (steppers != nullptr) cudaSafeCall(cudaFree(steppers));
        if (work != nullptr) cudaSafeCall(cudaFree(work));
        if (cwork != nullptr) cudaSafeCall(cudaFree(cwork));
        if (iwork != nullptr) cudaSafeCall(cudaFree(iwork));
    }
};

} // end of namespace cuda::ode::radau5

#endif // cuda_ode_radau5_stepper_cuh
// end of file
