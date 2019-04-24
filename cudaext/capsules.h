// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-2019 parasim inc
// (c) 2010-2019 california institute of technology
// all rights reserved
//
// Author(s): Lijun Zhu

#if !defined(cualtar_extensions_capsules_h)
#define cualtar_extensions_capsules_h

// get pyre.cuda capsules 
#include <pyre/cuda/capsules.h>


// capsules
namespace altar {
    namespace cuda {
        namespace extensions {
            // make alias of pyre cuda capsules
            namespace vector = pyre::extensions::cuda::vector;
            namespace matrix = pyre::extensions::cuda::matrix;
            namespace stream = pyre::extensions::cuda::stream;
        // add capsule name here
        }
    }
}

#endif

// end of file
