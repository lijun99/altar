// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-present parasim inc
// (c) 2010-present california institute of technology
// all rights reserved
//

#if !defined(altar_extensions_models_cudaseismic_capsules_h)
#define altar_extensions_models_cudaseismic_capsules_h

// cuda utilities
#include <pyre/cuda.h>
#include <pyre/cuda/capsules.h>

// capsules
namespace altar {
    namespace extensions {
        namespace models {
            namespace cudaseismic {
                    
                const char * const kgSmodel_capsule = "altar.models.cudaSkinematicg";
                const char * const kgDmodel_capsule = "altar.models.cudaDkinematicg";
            
            }
        }
    }
}

// types
namespace altar {
    namespace extensions {
        namespace models {
            namespace cudaseismic {
                namespace vector = pyre::extensions::cuda::vector;
                namespace matrix = pyre::extensions::cuda::matrix;
                namespace stream = pyre::extensions::cuda::stream;
            }
        }
    }
}

#endif

// end of file
