// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-present parasim inc
// (c) 2010-present california institute of technology
// all rights reserved
//
// hailiang zhang

#include <stdio.h>
#include "cudaKinematic_kernels.h"

namespace cudaKinematic_kernels {
    // copy the {cells} arrival times {T} of a sample into its snapshot {k}, if it keeps any; the
    // {threads} threads of the block share the copy, and wait for each other before and after
    template <typename TYPE>
    __device__ void keep(const TYPE * const T, TYPE * const snapshots, int k, int cells, int id, int threads)
    {
        if (snapshots == nullptr) return;
        __syncthreads();
        for (int c = id; c < cells; c += threads) snapshots[k*cells + c] = T[c];
        __syncthreads();
    }

    // the share of the gradient {g} of min(a, b) that goes to {a}: all of it, half on a tie, or none
    template <typename TYPE>
    __device__ TYPE dmin(TYPE a, TYPE b, TYPE g)
    {
        return a < b ? g : (a == b ? g/2 : TYPE(0));
    }
}

/// @par Main functionality
/// Seed the arrival times with straight ray times from the hypocenter within @b radius mesh cells,
/// rising steeply past it so the sweeps take over; continuous in the hypocenter
/// @param [in] gIdx the model index map that maps the indices in @b M matrix to the indices used in the codes (Dimension: parameters)
/// @param [in] gM the @b M matrix on GPU (Dimension: samples*parameters)
/// @param [in, out] gT0 the T0 values for each patch mesh point (Dimension: samples*((Nas+2)*Nmesh)*((Ndd+2)*Nmesh))
/// @param [in] radius the radius of the straight ray seeding, in mesh cells
/// @param [in] steepness how fast the seeds rise past {radius}
/// @param [in] it0 a large value to cap t0 (1.e6 as used in Sarah's code)
/// @note IT ALSO ASSUMES THE HYPO CENTER COORDINATES ORIGINATE FROM THE LEFT/BOTTOM PATCH CENTER OF THE FAULT PLANE
template <typename TYPE>
__global__ void
cudaKinematic_kernels::
initT0_batched(const size_t * const gIdx, const TYPE *const gM, TYPE * const gT0, const size_t Nparam,
    const size_t Nas, const size_t Ndd, const size_t Nmesh, const TYPE dsp,
    const TYPE radius, const TYPE steepness, const TYPE it0)
{
    // sample index
    int sample = blockIdx.x;
    // index of dip meshgrid
    int id_dip = threadIdx.y + blockIdx.y * blockDim.y;
    // index of strike meshgrid
    int id_strike = threadIdx.z + blockIdx.z * blockDim.z;
    // get dimension and sizes of mesh grids
    const int Nddf = (Ndd+2)*Nmesh;
    const int Nasf = (Nas+2)*Nmesh;
    if (id_dip >= Nddf || id_strike >= Nasf) return;
    const int Npatch = Nas*Ndd;
    // get the pointer for this sample, gM[samples, parameters]
    const TYPE * gM_sample = gM + sample*Nparam;
    // get the hypocenter from M/theta, shifted due to the extra padded edges
    TYPE hypo_strike = gM_sample[gIdx[4*Npatch]] + dsp*1.5;
    TYPE hypo_dip = gM_sample[gIdx[4*Npatch+1]] + dsp*1.5;
    // the distance of the mesh point from the hypocenter, and how far past the radius it is
    TYPE dspf = dsp/Nmesh;
    TYPE distance_strike = (id_strike+0.5)*dspf - hypo_strike;
    TYPE distance_dip = (id_dip+0.5)*dspf - hypo_dip;
    TYPE distance = sqrt(distance_strike*distance_strike + distance_dip*distance_dip);
    TYPE far = max(TYPE(0), distance - radius*dspf);
    // the rupture velocity of the patch of the mesh point, clamped to the fault
    int id_strike_patch = min(max(id_strike/(int)Nmesh - 1, 0), (int)Nas - 1);
    int id_dip_patch = min(max(id_dip/(int)Nmesh - 1, 0), (int)Ndd - 1);
    TYPE vr = gM_sample[gIdx[3*Npatch + id_strike_patch*Ndd + id_dip_patch]];
    // the seed, gT0[samples, Nasf, Nddf]
    gT0[sample*Nddf*Nasf + id_strike*Nddf + id_dip] = min((distance + steepness*far*far/dspf)/vr, it0);
}

/// @par Main functionality
/// Upwind device function used for cudaFastSweeping
/// @note see @c cudaFastSweeping function for detailed parameter description
template <typename TYPE>
__device__ TYPE
cudaKinematic_kernels::
upwind(const size_t * gIdx, const TYPE *const gT0, const int i, const int j,
    const size_t Nas, const size_t Ndd, const size_t Nmesh, const TYPE *const gM, const TYPE h)
{
    // i is id_strike; j is id_dip
    int Nasf = (Nas+2)*Nmesh;
    int Nddf = (Ndd+2)*Nmesh;
    int Npatch = Nas*Ndd;
    // indices for nearest neighbor mesh grids
    int i1, i2, j1, j2;
    i1=max(0,i-1);
    i2=min(Nasf-1,i+1);
    j1=max(0,j-1);
    j2=min(Nddf-1,j+1);

    // current arrival time
    TYPE u_old = gT0[i*Nddf +j];
    // arrival time of neighbors (take the smaller one)
    TYPE u_xmin = min(gT0[i1*Nddf+j], gT0[i2*Nddf+j]);
    TYPE u_ymin = min(gT0[i*Nddf+j1], gT0[i*Nddf+j2]);
    TYPE u_new;
    TYPE f;
    // find the rupture velocity
    // reuse i1, j1 for their patch index (not mesh grid index)
    i1=i/Nmesh-1;
    if (i1==-1) i1=0;
    if (i1>=Nas) i1=Nas-1;
    j1=j/Nmesh-1;
    if (j1==-1) j1=0;
    if (j1==Ndd) j1=Ndd-1;
    // get the inverse rupture velocity
    f = 1./gM[gIdx[3*Npatch+i1*Ndd+j1]]; // the index need to be shifted from gT0 to gM
    // compute the new arrival time propagating from neighbors
    if (fabs(u_xmin-u_ymin) >= f*h)
    {
        // big difference along dip and strike, use the smaller value
        u_new = min(u_xmin, u_ymin) + f*h;
    }
    else
    {
        // no big difference, use averaged method
        u_new = ( u_xmin + u_ymin + sqrt(2.*f*f*h*h-(u_xmin-u_ymin)*(u_xmin-u_ymin)) ) / 2.0;
        //u_new = ( u_xmin + u_ymin + sqrt( 2.*pow(f*h,2.0)-pow((u_xmin-u_ymin),2.0)) ) / 2.0;
    }
//printf("%d %d uold %f unew %f f: %f\n",i,j,u_old,u_new,f);
    return min(u_new, u_old);
}



/// @par Main functionality
/// Fast sweeping
/// @param [in] gIdx the model index map that maps the indices in @b M matrix to the indices used in the codes (Dimension: parameters)
/// @param [in] gM the @b M matrix on GPU (Dimension: parameters*samples, samples is the leading index)
/// @param [in, out] gT0 the T0 values for each patch mesh point (Dimension: ((Nas+2)*(Ndd+2)*Nmesh*Nmesh) * samples, samples is the leading index)
/// @param [in] Ns number of samples
/// @param [in] Ns_good number of good samples (pass the "verify" function test)
/// @param [in] Np number of patches
/// @param [in] Nas number of patches along strike
/// @param [in] Ndd number of patches down dip
/// @param [in] Nmesh number of mesh points per patch dimension used for fast sweeping
/// @param [in] h the distance between 2 adjacent mesh points
/// @param [out] gSnapshots the arrival times before the first sweep and after each one, for the adjoint (Dimension: samples*(4*iteration+1)*((Nas+2)*Nmesh)*((Ndd+2)*Nmesh)); none when null
/// @note IT ASSUMES NDD AS THE LEADING INDEX OF PARAMETERS<br>
/// IT ALSO ASSUMES THE HYPO CENTER COORINATES ORIGINATEF FROM THE LEFT/BOTTOM PATCH CENTER OF THE FAULT PLANE
/// @note
/// <pre>
/// the non-diagonal mesh grid will be expanded to a diagonal one with the longer dimension<br>
/// Nas
/// ^
/// |______________________
/// |0 (id)            9   |
/// |  1             8     |
/// |    2         7       |
/// |      3     6         |
/// |        4 5           |
/// |        4 5           |
/// |      3     6         |
/// |    2         7       |
/// |  1             8     |
/// |0 (id)            9   |
/// 0____________________________>  Ndd
/// </pre>
template <typename TYPE>
__global__ void
cudaKinematic_kernels::
fastSweeping_batched(const size_t * gIdx, const TYPE *const gM, TYPE *const gT0,
    const size_t Nparam, const size_t Ns_good, const size_t Nas, const size_t Ndd, const size_t Nmesh,
    const TYPE h, const size_t iteration, TYPE *const gSnapshots)
{
    // get the sample
    int sample = blockIdx.x;
    // if (sample>=Ns_good) return;

    const int Nddf = (Ndd+2)*Nmesh;
    const int Nasf = (Nas+2)*Nmesh;

    // get the data pointer for current sample
    const TYPE * gM_sample  = gM + sample*Nparam;
    TYPE * gT0_sample = gT0 + sample*Nddf*Nasf;

    // get the id along the diagonal of the "expanded" diagonal mesh
    const int id = threadIdx.x;
    if (id>=max(Nasf,Nddf)) return;

    // get the dimension of the "expanded" diagonal mesh; not blockDim.x, whose extra threads have returned
    const int Nf = max(Nasf, Nddf);

    // some local variables
    int i;
    int nas, ndd; // the moving mesh coordinate for the present mesh id

    // the snapshots of this sample, if any
    TYPE * gSnapshots_sample = gSnapshots ? gSnapshots + sample*(4*iteration+1)*Nddf*Nasf : nullptr;
    int snapshot = 0;
    keep(gT0_sample, gSnapshots_sample, snapshot++, Nddf*Nasf, id, Nf);

    // fast sweeping for a number of iterations (hardwared to be 1 iteration as in Sarah's code)
    for (int iter=0; iter<iteration; iter++)
    {
        // sweeping along direction (+Nas,+Ndd)
        //   ie. //for (i=0; i<Nasf; i++) for (j=0; j<Nddf; j++)
        //        gT0[(i*Nddf+j)*Ns + sample] = upwind(gIdx, gT0, i, j, Np, Nas, Ndd, Nmesh, Ns, sample, gM, h);
        // get the starting mesh coordinate for the present mesh id
        ndd = id - Nf/2;
        nas = Nf/2 - id;
        for (i=0; i<Nf; ++i)
        {
            // Upwind the present mesh
            if (nas>=0 && nas<Nasf && ndd>=0 && ndd<Nddf)
                gT0_sample[nas*Nddf+ndd] = upwind(gIdx, gT0_sample, nas, ndd, Nas, Ndd, Nmesh, gM_sample, h);
            __syncthreads();
            // increment ndd
            ++ndd;
            // Upwind the right of the present mesh
            if (nas>=0 && nas<Nasf && ndd>=0 && ndd<Nddf)
                gT0_sample[nas*Nddf+ndd] = upwind(gIdx, gT0_sample, nas, ndd, Nas, Ndd, Nmesh, gM_sample, h);
            __syncthreads();
            // increment nas
            ++nas;
        }
        keep(gT0_sample, gSnapshots_sample, snapshot++, Nddf*Nasf, id, Nf);
        // sweeping along direction (-Nas,+Ndd)
        //   ie. //for (i=Nasf-1; i>=0; i--) for (j=0; j<Nddf; j++)
        //     gT0_sample[nas*Nddf+ndd] = upwind(gIdx, gT0_sample, nas, ndd, Nas, Ndd, Nmesh, gM_sample, h);
        // get the starting mesh coordinate for the present mesh id
        ndd = id - Nf/2;
        nas = id + Nf/2;
        for (i=0; i<Nf; ++i)
        {
            // Upwind the present mesh
            if (nas>=0 && nas<Nasf && ndd>=0 && ndd<Nddf)
                gT0_sample[nas*Nddf+ndd] = upwind(gIdx, gT0_sample, nas, ndd, Nas, Ndd, Nmesh, gM_sample, h);
            __syncthreads();
            // increment ndd
            ++ndd;
            // Upwind the right of the present mesh
            if (nas>=0 && nas<Nasf && ndd>=0 && ndd<Nddf)
                gT0_sample[nas*Nddf+ndd] = upwind(gIdx, gT0_sample, nas, ndd, Nas, Ndd, Nmesh, gM_sample, h);
            __syncthreads();
            // decrement nas
            --nas;
        }
        keep(gT0_sample, gSnapshots_sample, snapshot++, Nddf*Nasf, id, Nf);
        // sweeping along direction (-Nas,-Ndd)
        //   ie. for (i=Nasf-1; i>=0; i--) for (j=Nddf-1; j>=0; j--) gT0[(i*Nddf+j)*Ns + sample] = upwind(gIdx, gT0, i, j, Np, Nas, Ndd, Nmesh, Ns, sample, gM, h);
        // get the starting mesh coordinate for the present mesh id
        ndd = id + Nf/2;
        nas = (Nf/2 - id) + (Nf-1);
        for (i=0; i<Nf; ++i)
        {
            // Upwind the present mesh
            if (nas>=0 && nas<Nasf && ndd>=0 && ndd<Nddf)
                gT0_sample[nas*Nddf+ndd] = upwind(gIdx, gT0_sample, nas, ndd, Nas, Ndd, Nmesh, gM_sample, h);
            __syncthreads();
            // decrement ndd
            --ndd;
            // Upwind the left of the present mesh
            if (nas>=0 && nas<Nasf && ndd>=0 && ndd<Nddf)
                gT0_sample[nas*Nddf+ndd] = upwind(gIdx, gT0_sample, nas, ndd, Nas, Ndd, Nmesh, gM_sample, h);
            __syncthreads();
            // decrement nas
            --nas;
        }
        keep(gT0_sample, gSnapshots_sample, snapshot++, Nddf*Nasf, id, Nf);
        // sweeping along direction (+Nas,-Ndd)
        //   ie. //for (i=0; i<Nasf; i++) for (j=Nddf-1; j>=0; j--) gT0[(i*Nddf+j)*Ns + sample] = upwind(gIdx, gT0, i, j, Np, Nas, Ndd, Nmesh, Ns, sample, gM, h);
        // get the starting mesh coordinate for the present mesh id
        ndd = id + Nf/2;
        nas = id - Nf/2;
        for (i=0; i<Nf; ++i)
        {
            // Upwind the present mesh
            if (nas>=0 && nas<Nasf && ndd>=0 && ndd<Nddf)
                gT0_sample[nas*Nddf+ndd] = upwind(gIdx, gT0_sample, nas, ndd, Nas, Ndd, Nmesh, gM_sample, h);
            __syncthreads();
            // decrement ndd
            --ndd;
            // Upwind the left of the present mesh
            if (nas>=0 && nas<Nasf && ndd>=0 && ndd<Nddf)
                gT0_sample[nas*Nddf+ndd] = upwind(gIdx, gT0_sample, nas, ndd, Nas, Ndd, Nmesh, gM_sample, h);
            __syncthreads();
            // increment nas
            ++nas;
        }
        keep(gT0_sample, gSnapshots_sample, snapshot++, Nddf*Nasf, id, Nf);
    }
}


/// @par Main functionality
/// interpolate T0 to TI0
/// @param [in] gT0 the T0 values for each patch mesh point (Dimension: ((Nas+2)*(Ndd+2)*Nmesh*Nmesh) * samples, samples is the leading index)
/// @param [in, out] gTI0 the T0 values for each interpolated point (Dimension: (Npatch*Npt*Npt) * samples, samples is the leading index)
/// @param [in] Ns number of samples
/// @param [in] Ns_good number of good samples (pass the "verify" function test)
/// @param [in] Nas number of patches along strike
/// @param [in] Ndd number of patches down dip
/// @param [in] Nmesh number of mesh points per patch dimension used for fast sweeping
/// @param [in] Npt_gi the number of source time functions for T0 interpolation along each dimension
/// @note A coordinate system is constructed with gT0[0] as the origin, and each patch length as Nmesh
/// The leading index is along Ndd direction
template <typename TYPE>
__global__ void
cudaKinematic_kernels::
interpolateT0_batched(const TYPE *const gT0, TYPE *const gTI0, const size_t Ns_good,
    const size_t Nas, const size_t Ndd, const size_t Nmesh, const size_t Npt_gi)
{
    int sample = threadIdx.x + blockIdx.x * blockDim.x;
    if (sample>=Ns_good) return;

    int Nddf = (Ndd+2)*Nmesh;
    int Nasf = (Nas+2)*Nmesh;

    const TYPE * gT0_sample = gT0 + sample*Nddf*Nasf;
    TYPE * gTI0_sample = gTI0 + sample*(Npt_gi*Nas)*(Npt_gi*Ndd);

    int i, j;
    TYPE x, y; // the absolute coordinate of a source point
    TYPE xr, yr; // the fraction part of the x/y
    int idx, idy; // the index of LEFT/BOTTOM gT0 point for a given source point
    TYPE offset = (TYPE(Nmesh)-0.5) + 0.5*TYPE(Nmesh)/TYPE(Npt_gi); // the starting coordinate of the first source point
    TYPE f11, f21, f12, f22;
    for (i=0; i<Ndd*Npt_gi; i++)
    {
        x = offset + TYPE(i)*TYPE(Nmesh)/TYPE(Npt_gi);
        xr = x-(TYPE)(int(x));
        idx = int(x);
        for (j=0; j<Nas*Npt_gi; j++)
        {
            y = offset + TYPE(j)*TYPE(Nmesh)/TYPE(Npt_gi);
            yr = y-(TYPE)(int(y));
            idy = int(y);

            f11 = gT0_sample[idx + idy*Nddf];
            f21 = gT0_sample[idx+1 + idy*Nddf];
            f12 = gT0_sample[idx + (idy+1)*Nddf];
            f22 = gT0_sample[(idx+1) + (idy+1)*Nddf];
            gTI0_sample[i + j*(Ndd*Npt_gi)]
                = f11*(1.0-xr)*(1.0-yr) + f21*xr*(1.0-yr) + f12*(1.0-xr)*yr + f22*xr*yr;
        }
    }
}


/// @par Main functionality
/// Cast M to M_big
/// @param [in] gIdx the model index map that maps the indices in @b M matrix to the indices used in the codes (Dimension: parameters)
/// @param [in] gM the @b M matrix on GPU (Dimension: parameters*samples, samples is the leading index)
/// @param [in] gTI0 the T0 values for each interpolated point (Dimension: (Npatch*Npt*Npt) * samples, samples is the leading index)
/// @param [in, out] gMb the @b Mb matrix on GPU (Dimension: (2*<c>Nt</c>*Npatch)*samples, samples is the leading index)
/// @param [in] gt0s the t0 values of small triangle source time funtion of all patches (Dimension: Npatch)
/// @param [in] dt the width of the small triangle source time function
/// @param [in] Ns number of samples
/// @param [in] Ns_good number of good samples (pass the "verify" function test)
/// @param [in] Nt number of small triangle source time functions
/// @param [in] Np number of patches
/// @param [in] Nas number of patches along strike
/// @param [in] Ndd number of patches down dip
/// @param [in] Npt_gi the number of source time functions for T0 interpolation along each dimension
template <typename TYPE>
__global__ void
cudaKinematic_kernels::
castBigM_batched(const size_t * gIdx, const TYPE *const gM, const TYPE *const gTI0, TYPE *const gMb, const TYPE *const gt0s,
    const TYPE dt, const size_t Nparam, const size_t Nt, const size_t Nas, const size_t Ndd, const size_t Npt_gi)
{
    int sample = blockIdx.x;
    int patch = threadIdx.y + blockIdx.y * blockDim.y;

    int Npatch = Nas*Ndd;
    if (patch >= Npatch) return;

    //if (Check_badmove(gM, sample, Ns, Np, para_low, para_high, vr_low, vr_high, tr_low, tr_high, h0s_low, h0s_high, h0d_low, h0d_high)) {return;}

    const TYPE * gM_sample  = gM + sample*Nparam;  //gM [samples][parameters] leading dimension on right
    const TYPE * gTI0_sample = gTI0 + sample*(Npt_gi*Nas)*(Npt_gi*Ndd); // gTI0[samples][Nas][Npt][Ndd][Npt]
    TYPE * gMb_sample = gMb + sample*Nt*2*Nas*Ndd;
        //gMb [samples][Nt][2(strike,dip slips)][Nas][Ndd]

    // for single (current) sample

    //strike slip
    TYPE ss = gM_sample[gIdx[patch]];
    // dip slip
    TYPE sd = gM_sample[gIdx[patch+Npatch]];
    // rupture time
    TYPE Tr = gM_sample[gIdx[2*Npatch+patch]];
    TYPE TI0; // arrival time for integration
    TYPE st0 = gt0s[patch]; //starting time
    TYPE sdt = dt;
    TYPE t0;
    TYPE hTr=Tr/2.0;
    TYPE c; //
    // get the patch id along dip/strike
    int idx=patch%Ndd, idy=patch/Ndd; // the patch index
    int  idx_ti, idy_ti; // the gTI0 index
    for (int i=0; i<Nt; i++)
    {
        t0=st0+sdt*i; //current time
        c=0.0;
        // loop over Npt to integrate
        for (idx_ti=idx*Npt_gi; idx_ti<(idx+1)*Npt_gi; idx_ti++)
        {
            for (idy_ti=idy*Npt_gi; idy_ti<(idy+1)*Npt_gi; idy_ti++)
            {
                TI0 = gTI0_sample[idx_ti + idy_ti*(Ndd*Npt_gi)]+hTr;
                if (t0<=(TI0-hTr)||t0>=(TI0+hTr)) continue;
                else c += (1.0-abs(TI0-t0)/hTr)*(1.0/hTr)/(Npt_gi*Npt_gi);
            }
        }
        gMb_sample[i*(2*Npatch)+patch] = c*ss;
        gMb_sample[i*(2*Npatch)+Npatch+patch] = c*sd;
    }
}


/// @par Main functionality
/// The adjoint of castBigM: from @b gdMb, the gradient of the log likelihood with respect to Mb,
/// the gradients with respect to the strike and dip slips and the rise times of each patch, into
/// their columns of @b gGrad, and with respect to the arrival times at its source points, @b gdTI0
/// @note one thread per patch of a sample, as castBigM; ties of |.| at 0 contribute nothing, as in jax
template <typename TYPE>
__global__ void
cudaKinematic_kernels::
castBigM_adjoint_batched(const size_t * gIdx, const TYPE *const gM, const TYPE *const gTI0, const TYPE *const gdMb,
    const TYPE *const gt0s, const TYPE dt, const size_t Nparam, const size_t Nt, const size_t Nas, const size_t Ndd,
    const size_t Npt_gi, TYPE *const gGrad, TYPE *const gdTI0)
{
    int sample = blockIdx.x;
    int patch = threadIdx.y + blockIdx.y * blockDim.y;
    int Npatch = Nas*Ndd;
    if (patch >= Npatch) return;

    const TYPE * gM_sample  = gM + sample*Nparam;
    const TYPE * gTI0_sample = gTI0 + sample*(Npt_gi*Nas)*(Npt_gi*Ndd);
    const TYPE * gdMb_sample = gdMb + sample*Nt*2*Nas*Ndd;
    TYPE * gdTI0_sample = gdTI0 + sample*(Npt_gi*Nas)*(Npt_gi*Ndd);

    TYPE ss = gM_sample[gIdx[patch]];
    TYPE sd = gM_sample[gIdx[patch+Npatch]];
    TYPE Tr = gM_sample[gIdx[2*Npatch+patch]];
    TYPE st0 = gt0s[patch];
    TYPE hTr = Tr/2.0;
    TYPE n2 = Npt_gi*Npt_gi;
    int idx = patch%Ndd, idy = patch/Ndd;

    // the gradients with respect to the slips and the half rise time
    TYPE gss = 0, gsd = 0, ghTr = 0;
    for (int idx_ti=idx*Npt_gi; idx_ti<(idx+1)*Npt_gi; idx_ti++) {
        for (int idy_ti=idy*Npt_gi; idy_ti<(idy+1)*Npt_gi; idy_ti++) {
            gdTI0_sample[idx_ti + idy_ti*(Ndd*Npt_gi)] = 0;
        }
    }
    for (int i=0; i<Nt; i++) {
        TYPE t0 = st0 + dt*i;
        TYPE dss = gdMb_sample[i*(2*Npatch)+patch];
        TYPE dsd = gdMb_sample[i*(2*Npatch)+Npatch+patch];
        // Mb = c ss, c sd: the gradient with respect to c
        TYPE gc = dss*ss + dsd*sd;
        TYPE c = 0;
        for (int idx_ti=idx*Npt_gi; idx_ti<(idx+1)*Npt_gi; idx_ti++) {
            for (int idy_ti=idy*Npt_gi; idy_ti<(idy+1)*Npt_gi; idy_ti++) {
                int k = idx_ti + idy_ti*(Ndd*Npt_gi);
                TYPE TI0 = gTI0_sample[k] + hTr;
                if (t0<=(TI0-hTr)||t0>=(TI0+hTr)) continue;
                TYPE a = TI0 - t0;
                TYPE absa = abs(a);
                TYPE sign = TYPE((a > 0) - (a < 0));
                c += (1.0-absa/hTr)*(1.0/hTr)/n2;
                // v = (1 - |a|/h)/h/n2, with a = TI0 + h - t
                TYPE dv_da = -sign/(hTr*hTr*n2);
                TYPE dv_dh = (2*absa/hTr - 1)/(hTr*hTr*n2);
                gdTI0_sample[k] += gc*dv_da;
                ghTr += gc*(dv_dh + dv_da);
            }
        }
        gss += dss*c;
        gsd += dsd*c;
    }
    TYPE * gGrad_sample = gGrad + sample*Nparam;
    gGrad_sample[gIdx[patch]] = gss;
    gGrad_sample[gIdx[patch+Npatch]] = gsd;
    gGrad_sample[gIdx[2*Npatch+patch]] = ghTr/2.0;
}


/// @par Main functionality
/// The adjoint of interpolateT0: scatter the gradients with respect to the arrival times at the
/// source points, @b gdTI0, onto the mesh, @b gdT0, which must be zeroed first
template <typename TYPE>
__global__ void
cudaKinematic_kernels::
interpolateT0_adjoint_batched(const TYPE *const gdTI0, TYPE *const gdT0, const size_t Ns_good,
    const size_t Nas, const size_t Ndd, const size_t Nmesh, const size_t Npt_gi)
{
    int sample = threadIdx.x + blockIdx.x * blockDim.x;
    if (sample>=Ns_good) return;

    int Nddf = (Ndd+2)*Nmesh;
    int Nasf = (Nas+2)*Nmesh;
    TYPE * gdT0_sample = gdT0 + sample*Nddf*Nasf;
    const TYPE * gdTI0_sample = gdTI0 + sample*(Npt_gi*Nas)*(Npt_gi*Ndd);

    TYPE offset = (TYPE(Nmesh)-0.5) + 0.5*TYPE(Nmesh)/TYPE(Npt_gi);
    for (int i=0; i<Ndd*Npt_gi; i++) {
        TYPE x = offset + TYPE(i)*TYPE(Nmesh)/TYPE(Npt_gi);
        TYPE xr = x-(TYPE)(int(x));
        int idx = int(x);
        for (int j=0; j<Nas*Npt_gi; j++) {
            TYPE y = offset + TYPE(j)*TYPE(Nmesh)/TYPE(Npt_gi);
            TYPE yr = y-(TYPE)(int(y));
            int idy = int(y);
            TYPE g = gdTI0_sample[i + j*(Ndd*Npt_gi)];
            gdT0_sample[idx + idy*Nddf] += g*(1.0-xr)*(1.0-yr);
            gdT0_sample[idx+1 + idy*Nddf] += g*xr*(1.0-yr);
            gdT0_sample[idx + (idy+1)*Nddf] += g*(1.0-xr)*yr;
            gdT0_sample[(idx+1) + (idy+1)*Nddf] += g*xr*yr;
        }
    }
}


namespace cudaKinematic_kernels {
    // the order in which sweep {s} updates the mesh point {i}, {j}: a point reads the neighbors it
    // precedes as they were before the sweep, and the ones that precede it as they are after it
    __device__ inline int order(int s, int i, int j)
    {
        switch (s) {
            case 0: return i + j;
            case 1: return j - i;
            case 2: return -(i + j);
            default: return i - j;
        }
    }
}


/// @par Main functionality
/// The adjoint of the fast sweeping and of the seeding: from @b gdT0, the gradient with respect to
/// the swept arrival times, the gradients with respect to the rupture velocities and the hypocenter,
/// into their columns of @b gGrad; the sweeps run backwards, the inputs of each update rebuilt from
/// the snapshots the forward sweeps kept
/// @note one block per sample, threads as in fastSweeping_batched; @b gdT0 is overwritten
template <typename TYPE>
__global__ void
cudaKinematic_kernels::
fastSweeping_adjoint_batched(const size_t * gIdx, const TYPE *const gM, const TYPE *const gSnapshots,
    TYPE *const gdT0, const size_t Nparam, const size_t Nas, const size_t Ndd, const size_t Nmesh,
    const TYPE dsp, const TYPE radius, const TYPE steepness, const TYPE it0, const size_t iteration,
    TYPE *const gGrad)
{
    // the gradients with respect to the rupture velocity of each patch, and to the hypocenter
    extern __shared__ unsigned char shared[];
    TYPE * gvr = reinterpret_cast<TYPE *>(shared);

    int sample = blockIdx.x;
    const int Nddf = (Ndd+2)*Nmesh;
    const int Nasf = (Nas+2)*Nmesh;
    const int cells = Nddf*Nasf;
    const int Npatch = Nas*Ndd;
    const int Nf = max(Nasf, Nddf);
    const int id = threadIdx.x;
    const bool active = id < Nf;
    const TYPE h = dsp/Nmesh;

    const TYPE * gM_sample = gM + sample*Nparam;
    const TYPE * snapshots = gSnapshots + sample*(4*iteration+1)*cells;
    TYPE * lambda = gdT0 + sample*cells;

    for (int p = id; p < Npatch+2; p += blockDim.x) gvr[p] = 0;
    __syncthreads();

    // the sweeps, backwards: the half-steps of each, in reverse
    const int half = Nf/2;
    for (int sweep = 4*iteration-1; sweep >= 0; --sweep) {
        const int s = sweep % 4;
        const TYPE * before = snapshots + sweep*cells;
        const TYPE * after = before + cells;
        // where the thread starts, and how it moves, as in fastSweeping_batched
        int nas0, ndd0, dnas, dndd;
        switch (s) {
            case 0: nas0 = half - id; ndd0 = id - half; dnas = 1; dndd = 1; break;
            case 1: nas0 = id + half; ndd0 = id - half; dnas = -1; dndd = 1; break;
            case 2: nas0 = (half - id) + (Nf-1); ndd0 = id + half; dnas = -1; dndd = -1; break;
            default: nas0 = id - half; ndd0 = id + half; dnas = 1; dndd = -1; break;
        }
        for (int q = 2*Nf-1; q >= 0; --q) {
            const int k = q/2;
            const int i = nas0 + k*dnas;
            const int j = ndd0 + (k + (q % 2))*dndd;
            if (active && i>=0 && i<Nasf && j>=0 && j<Nddf) {
                const int cell = i*Nddf + j;
                const int here = cudaKinematic_kernels::order(s, i, j);
                // the neighbors, and what the update read of each
                int n[4] = { max(0,i-1)*Nddf + j, min(Nasf-1,i+1)*Nddf + j, i*Nddf + max(0,j-1), i*Nddf + min(Nddf-1,j+1) };
                int ni[4] = { max(0,i-1), min(Nasf-1,i+1), i, i };
                int nj[4] = { j, j, max(0,j-1), min(Nddf-1,j+1) };
                TYPE u[4];
                for (int m = 0; m < 4; ++m) {
                    u[m] = n[m] == cell ? before[cell]
                        : (cudaKinematic_kernels::order(s, ni[m], nj[m]) < here ? after[n[m]] : before[n[m]]);
                }
                const TYPE u_old = before[cell];
                const TYPE ux = min(u[0], u[1]);
                const TYPE uy = min(u[2], u[3]);
                // the rupture velocity, as in upwind
                int pi = i/Nmesh - 1; pi = pi < 0 ? 0 : (pi >= (int)Nas ? Nas-1 : pi);
                int pj = j/Nmesh - 1; pj = pj < 0 ? 0 : (pj >= (int)Ndd ? Ndd-1 : pj);
                const int patch = pi*Ndd + pj;
                const TYPE vr = gM_sample[gIdx[3*Npatch + patch]];
                const TYPE f = 1./vr;
                const TYPE fh = f*h;
                const TYPE d = ux - uy;
                const bool apart = fabs(d) >= fh;
                const TYPE root = apart ? TYPE(1) : sqrt(2.*f*f*h*h - d*d);
                const TYPE u_new = apart ? min(ux, uy) + fh : (ux + uy + root)/2.0;
                // the gradient with respect to the updated value, split between the update and the old value
                const TYPE g = lambda[cell];
                const TYPE g_new = cudaKinematic_kernels::dmin(u_new, u_old, g);
                TYPE gu[4] = {0, 0, 0, 0};
                TYPE gux, guy, gfh;
                if (apart) {
                    gux = cudaKinematic_kernels::dmin(ux, uy, g_new);
                    guy = cudaKinematic_kernels::dmin(uy, ux, g_new);
                    gfh = g_new;
                } else {
                    gux = g_new*0.5*(1 - d/root);
                    guy = g_new*0.5*(1 + d/root);
                    gfh = g_new*fh/root;
                }
                gu[0] = cudaKinematic_kernels::dmin(u[0], u[1], gux);
                gu[1] = cudaKinematic_kernels::dmin(u[1], u[0], gux);
                gu[2] = cudaKinematic_kernels::dmin(u[2], u[3], guy);
                gu[3] = cudaKinematic_kernels::dmin(u[3], u[2], guy);
                // the cell now carries the gradient with respect to its value before the update
                TYPE g_old = cudaKinematic_kernels::dmin(u_old, u_new, g);
                for (int m = 0; m < 4; ++m) {
                    if (n[m] == cell) g_old += gu[m];
                    else atomicAdd(lambda + n[m], gu[m]);
                }
                lambda[cell] = g_old;
                // fh = h/vr
                atomicAdd(gvr + patch, -gfh*h/(vr*vr));
            }
            __syncthreads();
        }
    }

    // the seeding: T = min((distance + steepness far^2/h)/vr, it0), far = max(0, distance - radius h)
    const TYPE hypo_strike = gM_sample[gIdx[4*Npatch]] + dsp*1.5;
    const TYPE hypo_dip = gM_sample[gIdx[4*Npatch+1]] + dsp*1.5;
    for (int cell = id; cell < cells; cell += blockDim.x) {
        const int i = cell/Nddf, j = cell%Nddf;
        const TYPE ds = (i+0.5)*h - hypo_strike;
        const TYPE dd = (j+0.5)*h - hypo_dip;
        const TYPE distance = sqrt(ds*ds + dd*dd);
        const TYPE excess = distance - radius*h;
        const TYPE far = max(TYPE(0), excess);
        int pi = min(max(i/(int)Nmesh - 1, 0), (int)Nas - 1);
        int pj = min(max(j/(int)Nmesh - 1, 0), (int)Ndd - 1);
        const int patch = pi*Ndd + pj;
        const TYPE vr = gM_sample[gIdx[3*Npatch + patch]];
        const TYPE travel = (distance + steepness*far*far/h)/vr;
        const TYPE g = cudaKinematic_kernels::dmin(travel, it0, lambda[cell]);
        if (g == 0) continue;
        atomicAdd(gvr + patch, -g*travel/vr);
        // the distance, through the hypocenter; none at the hypocenter itself
        const TYPE dfar = excess > 0 ? TYPE(1) : (excess == 0 ? TYPE(0.5) : TYPE(0));
        const TYPE gdistance = g*(1 + 2*steepness*far*dfar/h)/vr;
        if (distance > 0) {
            atomicAdd(gvr + Npatch, -gdistance*ds/distance);
            atomicAdd(gvr + Npatch + 1, -gdistance*dd/distance);
        }
    }
    __syncthreads();

    TYPE * gGrad_sample = gGrad + sample*Nparam;
    for (int p = id; p < Npatch; p += blockDim.x) gGrad_sample[gIdx[3*Npatch + p]] = gvr[p];
    if (id == 0) {
        gGrad_sample[gIdx[4*Npatch]] = gvr[Npatch];
        gGrad_sample[gIdx[4*Npatch+1]] = gvr[Npatch+1];
    }
}


//end of file
