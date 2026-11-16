/*
 *  This is an example of an ode function required by the package
 *
 */

#ifndef __mytestode_cuh__
#define __mytestode_cuh__

// an example ode function
// please follow(copy) this example for naming conventions
template <class T>
struct __ALIGNED__ MyTestOde {
    // required parameters, keep their names
    // @note each system is defined by #patches and each patch with #units
    // @note each patch is processed by one thread
    int patches; // number of patches per system
    int units;   // number of units per patch
    int system_size; // patches*units
    int systems; // total systems/samples to be processed

    // other custom parameters
    // all these parameters need to set inside this structure
    T * alpha;

    // ode function for a given patch f= dy/dt = f(t, y) = exp(-alpha[sys]*t)
    __device__ __forceinline__ void dydt(const int system_id, const int patch_id, const T t, const T* y, T* f)
    {
        // an example of ode function, different system has a different prefactor alpha
        for (auto unit_id =0; unit_id < units; unit_id++)
            f[patch_id+unit_id*patches] = exp(-alpha[system_id]*t);
        return;
    };

    // ode function called when solving a system with a thread block
    __device__ __forceinline__  void dydt_block(const cg::thread_block& cta, const int system_id, const T t, const T* y0, T* f)
    {
        for(int patch_id = cta.thread_rank(); patch_id<patches; patch_id+=cta.size())
            dydt(system_id, patch_id, t, y0, f);
        // if(cta.thread_rank()==0)
        //    printf("test dydt %g\n", f[0]);
    };

    // you also add other methods to set custom parameters e.g.,
    void init_parameters()
    {
        // to set up alpha
        cudaMallocManaged(&alpha, systems*sizeof(T));
        for(auto i=0; i<systems; i++)
            alpha[i] = (T)(i+1)/systems;
    };

    // constructor
    MyTestOde(const int p=1, const int u=1, const int sys=1)
        : patches(p), units(u), systems(sys), system_size(p*u)
    {
        init_parameters();
    };
};

#endif //__mytestode_cuh__
// end of file