// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Hailiang Zhang, Lijun Zhu
// california institute of technology
// (c) 2010-2019  all rights reserved
//


#include <stdio.h>
#include "cudaKinematicG_kernels.h"

/// @par Main functionality
/// Initialize T0 to be Distances!!!
/// @param [in] gIdx the model index map that maps the indices in @b M matrix to the indices used in the codes (Dimension: parameters)
/// @param [in] gM the @b M matrix on GPU (Dimension: parameters*samples, samples is the leading index)
/// @param [in] Ns number of samples
/// @param [in] Ns_good number of good samples (pass the "verify" function test)
/// @param [in] Nas number of patches along strike
/// @param [in] Ndd number of patches down dip
/// @param [in] Nmesh number of mesh points per patch dimension used for fast sweeping
/// @param [in] dsp the length of each patch dimension
/// @param [in, out] gT0 the T0 values for each patch mesh point (Dimension: ((Nas+2)*(Ndd+2)*Nmesh*Nmesh) * samples, samples is the leading index)
/// @param [in] it0 a large value to initialize t0 (1.e6 as used in Sarah's code)


// set T0 for each sample
template<typename TYPE>
__device__ void
cudaKinematicG_kernels::
initT0(TYPE * const gT0, const size_t Nddf, const size_t Nasf, TYPE dspf, TYPE hypo_dip, TYPE hypo_strike, TYPE it0)
{
    // index of dip meshgrid
    int id_dip = threadIdx.x + blockIdx.x * blockDim.x;
    // index of strike meshgrid
    int id_strike = threadIdx.y + blockIdx.y * blockDim.y;
    // check the meshgrid is within range
    if(id_dip >= Nddf || id_strike >= Nasf) return;
    // meshgrid index in 2d strike(Nasf)xdip (Nddf) grids   
    int id_mesh = id_strike*Nddf + id_dip;
    // find the meshgrid of hypocenter
    int id_hypo_strike = int(hypo_strike/dspf);
    int id_hypo_dip = int(hypo_dip/dspf);
    // if meshgrid is not among the nearest 4x4 grids close to hypocenter
    // Comment by Lijun: why 4x4? 2x2 seems to be enough
    if (id_strike>id_hypo_strike-2 && id_strike<id_hypo_strike+2
        && id_dip > id_hypo_dip-2 && id_dip<id_hypo_dip+2) {
       // distance to hypocenter
       TYPE distance_strike =  (id_strike+0.5)*dspf- hypo_strike;
       TYPE distance_dip = (id_dip+0.5)*dspf - hypo_dip;
       // set the distance to time
       gT0[id_mesh] = sqrt(distance_strike*distance_strike+distance_dip*distance_dip);
    }
    else { // set a large number for nearest 4 grids
        gT0[id_mesh] = it0;
    }
}

template <typename TYPE>
__global__ void
cudaKinematicG_kernels::
initT0_batched(const size_t * const gIdx, const TYPE *const gM, TYPE * const gT0, const size_t Nparam,
    const size_t Nas, const size_t Ndd, const size_t Nmesh, const TYPE dsp, const TYPE it0)
{
    // sample index 
    int sample = blockIdx.z;
    // get the pointer for this sample, gM[samples, parameters]
    const TYPE * gM_sample = gM + sample*Nparam; 
    // get the hypocenter from M/theta
    TYPE hypo_strike = gM_sample[gIdx[4*Nas*Ndd]] + dsp*1.5; //shifted due to the extra padded edges
    TYPE hypo_dip = gM_sample[gIdx[4*Nas*Ndd+1]] + dsp*1.5;
    // get dimension and sizes of mesh grids
    int Nddf = (Ndd+2)*Nmesh; // x
    int Nasf = (Nas+2)*Nmesh; // y
    TYPE dspf = dsp/Nmesh; // size of meshgrid
    // get the gT0 pointer for this sample gT0[samples, Nasf, Nddf]  
    TYPE * gT0_sample = gT0 + sample*Nddf*Nasf;
    // call routine for one sample
    initT0(gT0_sample, Nddf, Nasf, dspf, hypo_dip, hypo_strike, it0);
    // all done
}

/// @par Main functionality
/// device function used for cudaSetT0 to set the 4 closest points from hypo center
/// @note see @c cudaSetT0 function for detailed parameter description

// setT0 for one sample
template <typename TYPE>
__device__ void
cudaKinematicG_kernels::
setT0hypo(const size_t * gIdx, const TYPE *const gM, TYPE *const gT0,
    const size_t Nas, const size_t Ndd, const size_t Nmesh, const TYPE dsp, const TYPE it0)
{

    // find the hypo cenceter meshgrid
    TYPE hypo_strike = gM[gIdx[4*Nas*Ndd]] + dsp*1.5; // shift coordinate the the left/bottom corner of the EXPANDED fault surf
    TYPE hypo_dip = gM[gIdx[4*Nas*Ndd+1]] + dsp*1.5; // shift coordinate the the left/bottom corner of the EXPANED fault surf
    TYPE dspf = dsp/Nmesh;
    int id_hypo_strike = int(hypo_strike/dspf);
    int id_hypo_dip = int(hypo_dip/dspf);

    int Nddf = (Ndd+2)*Nmesh;
    
    // search the 4x4 window near the hypocenter for nearest 4 points
    TYPE previous_min_distance, min_distance = 0.0, current_distance;
    int iD[4]; // recording the meshgrid index of nearest 4 points
    for (int n=0; n<4; ++n)
    {   
        previous_min_distance = min_distance;
        min_distance = it0;
        for (int i = id_hypo_strike-2; i<=id_hypo_strike+2; ++i)
        {
            for (int j = id_hypo_dip-2; j<=id_hypo_dip+2; ++j)
            {
                // get the grid index
                int id_mesh = i*Nddf + j;
                // get the distance to hypocenter
                current_distance = gT0[id_mesh];
                if (current_distance>previous_min_distance && current_distance<min_distance)
                {
                    // record the new grid with shorter distance
                    min_distance = current_distance;
                    iD[n] = id_mesh;
                }
            }
        }
    }

    // (re)set the arrival time for these 4x4 grids
    bool match; 
    for (int i = id_hypo_strike-2; i<=id_hypo_strike+2; ++i)
    {
        for (int j = id_hypo_dip-2; j<=id_hypo_dip+2; ++j)
        {
            // get the grid index
            int id_mesh = i*Nddf + j;
            match = false;
            // search the record of nearest 4 points for a match
            for (int n=0; n<4; ++n) {
                if(iD[n]==id_mesh) { // a match
                    // find the patch (not mesh grid) index for rupture velocity
                    int id_strike_patch = i/Nmesh -1;
                    int id_dip_patch = j/Nmesh -1;
                    // reset index if out of boundaries; not likely
                    if(id_strike_patch <0) id_strike_patch =0;
                    else if (id_strike_patch >= Nas) id_strike_patch = Nas-1;
                    if(id_dip_patch <0) id_dip_patch =0;
                    else if (id_dip_patch >= Nas) id_dip_patch = Nas-1;
                    // get the patch index in flattened notation
                    int id_patch = id_strike_patch*Ndd + id_dip_patch;
                    // get the rupture velocity    
                    TYPE vr = gM[gIdx[3*Nas*Ndd+id_patch]];
                    // set the time: distance/vr 
                    gT0[id_mesh] /= vr;
                    match = true;
                }
            }
            // not the nearest 4 points, set time to a large number
            if(!match) gT0[id_mesh] = it0;
        }
    }
    // all done
}


/// @par Main functionality
/// Set t0 for the 4 patches closest to hypo center<br>
/// Set all other T0s to be a large number (it0)<br>
/// @param [in] gIdx the model index map that maps the indices in @b M matrix to the indices used in the codes (Dimension: parameters)
/// @param [in] gM the @b M matrix on GPU (Dimension: parameters*samples, samples is the leading index)
/// @param [in, out] gT0 the T0 values for each patch mesh point (Dimension: ((Nas+2)*(Ndd+2)*Nmesh*Nmesh) * samples, samples is the leading index)
/// @param [in] Ns number of samples
/// @param [in] Ns_good number of good samples (pass the "verify" function test)
/// @param [in] Np number of patches
/// @param [in] Nas number of patches along strike
/// @param [in] Ndd number of patches down dip
/// @param [in] dsp the length of each patch dimension
/// @param [in] Nmesh number of mesh points per patch dimension used for fast sweeping
/// @param [in] it0 a large value to initialize t0 (1.e6 as used in Sarah's code)
/// @note IT ASSUMES NDD AS THE LEADING INDEX OF PARAMETERS<br>
/// IT ALSO ASSUMES THE HYPO CENTER COORINATES ORIGINATEF FROM THE LEFT/BOTTOM PATCH CENTER OF THE FAULT PLANE
template <typename TYPE>
__global__ void
cudaKinematicG_kernels::
setT0_batched(const size_t * gIdx, const TYPE *const gM, TYPE *const gT0, const size_t Nparam,
    const size_t Ns_good, const size_t Nas, const size_t Ndd, const size_t Nmesh, const TYPE dsp, const TYPE it0)
{
    int sample = threadIdx.x + blockIdx.x * blockDim.x;
    if (sample>=Ns_good) return;

    // get the pointer of M/theta for current sample
    const TYPE * gM_sample = gM + sample*Nparam;
    // get the pointer of gT0 for current sample
    TYPE * gT0_sample = gT0 + sample*(Nas+2)*Nmesh*(Ndd+2)*Nmesh;
    // set arrival time (for 4 nearest points/mesh grids close to hypocenter)for each sample
    setT0hypo(gIdx, gM_sample, gT0_sample, Nas, Ndd, Nmesh, dsp, it0);
}


/// @par Main functionality
/// Upwind device function used for cudaFastSweeping
/// @note see @c cudaFastSweeping function for detailed parameter description
template <typename TYPE>
__device__ TYPE
cudaKinematicG_kernels::
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
cudaKinematicG_kernels::
fastSweeping_batched(const size_t * gIdx, const TYPE *const gM, TYPE *const gT0,
    const size_t Nparam, const size_t Ns_good, const size_t Nas, const size_t Ndd, const size_t Nmesh,
    const TYPE h, const size_t iteration)
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

    // get the dimension of the "expanded" diagonal mesh
    const int Nf = blockDim.x;

    // some local variables
    int i;
    int nas, ndd; // the moving mesh coordinate for the present mesh id

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
cudaKinematicG_kernels::
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
cudaKinematicG_kernels::
castBigM_batched(const size_t * gIdx, const TYPE *const gM, const TYPE *const gTI0, TYPE *const gMb, const TYPE *const gt0s,
    const TYPE dt, const size_t Nparam, const size_t Nt, const size_t Nas, const size_t Ndd, const size_t Npt_gi)
{
    int sample = blockIdx.z;
    int patch = threadIdx.x + blockIdx.x * blockDim.x;

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


//end of file

