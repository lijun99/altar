# -*- python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#


# externals
import numpy
# the package
import altar
import altar.cuda
from altar.cuda import nvrtc


# the kernels, in CUDA C++, compiled by NVRTC the first time they are needed
SOURCE = r"""
// prediction[s, o] = slope * x[o] + intercept, one thread per observation of a sample
template <typename real>
__global__ void predictions(const real * theta, const real * x, real * prediction,
                            int samples, int parameters, int observations, int slope, int intercept)
{
    int s = blockIdx.y;
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= samples || o >= observations) return;
    const real * sample = theta + s * parameters;
    prediction[s * observations + o] = sample[slope] * x[o] + sample[intercept];
}

// the gradient of the log likelihood, -sum_o weight[o] r[o] dr[o]/dtheta, with the residual
// r = prediction - data, one block per sample; the block size must be a power of two
template <typename real>
__global__ void gradients(const real * theta, const real * x, const real * data,
                          const real * weight, real * grad,
                          int parameters, int observations, int slope, int intercept)
{
    extern __shared__ double sums[];
    const real * sample = theta + blockIdx.x * parameters;
    const real a = sample[slope], b = sample[intercept];
    // each thread sums a stride of the observations
    double dslope = 0, dintercept = 0;
    for (int o = threadIdx.x; o < observations; o += blockDim.x) {
        double w = weight[o] * (a * x[o] + b - data[o]);
        dslope += w * x[o];
        dintercept += w;
    }
    // and the block adds them up
    int t = threadIdx.x, n = blockDim.x;
    sums[t] = dslope;
    sums[n + t] = dintercept;
    __syncthreads();
    for (int half = n / 2; half > 0; half /= 2) {
        if (t < half) {
            sums[t] += sums[t + half];
            sums[n + t] += sums[n + t + half];
        }
        __syncthreads();
    }
    if (t == 0) {
        real * g = grad + blockIdx.x * parameters;
        g[slope] = -sums[0];
        g[intercept] = -sums[n];
    }
}
"""

# threads per block
THREADS = 256


# declaration
class NVRTC:
    """
    The cuda strategy, with kernels in CUDA C++ compiled at run time by NVRTC
    """


    def initialize(self, model):
        """
        Compile my kernels for the working precision and upload the x
        """
        self.model = model
        real = nvrtc.real[model.precision]
        self.names = (f"predictions<{real}>", f"gradients<{real}>")
        self.kernels = nvrtc.kernels(SOURCE, self.names)
        self.x = altar.cuda.vector(source=model.x, dtype=model.precision)
        # all done
        return self


    def forward_model_batched(self, theta, prediction, batch):
        """
        Fill the first {batch} rows of {prediction} with the predicted y of {theta}
        """
        model = self.model
        observations = self.x.shape[0]
        nvrtc.launch(self.kernels[self.names[0]],
                     (nvrtc.blocks(observations, THREADS), batch), THREADS,
                     theta, self.x, prediction,
                     numpy.int32(batch), numpy.int32(theta.shape[1]), numpy.int32(observations),
                     numpy.int32(model.slopeIdx), numpy.int32(model.interceptIdx))
        # all done
        return self


    def gradient(self, theta, data, weight, gradient, batch):
        """
        Fill the first {batch} rows of {gradient} with the gradient of the log likelihood of
        the {data}, whose observations have the inverse variances {weight}
        """
        model = self.model
        nvrtc.launch(self.kernels[self.names[1]], batch, THREADS,
                     theta, self.x, data, weight, gradient,
                     numpy.int32(theta.shape[1]), numpy.int32(self.x.shape[0]),
                     numpy.int32(model.slopeIdx), numpy.int32(model.interceptIdx),
                     shared=2 * THREADS * 8)
        # all done
        return self


    # private data
    model = None
    names = None # the names of my kernels, in the working precision
    kernels = None # and the kernels, by name
    x = None # the x of each observation, on the device


# end of file
