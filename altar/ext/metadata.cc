// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
//
// (c) 2013-present parasim inc
// (c) 2010-present california institute of technology
// all rights reserved
//


// external dependencies
#include "external.h"
// namespace setup
#include "forward.h"


void
altar::py::metadata(py::module & m)
{
    m.def(
        "copyright",
        []() -> std::string {
            return "altar.beta: (c) 2013-present ParaSim Inc; 2010-present California Institute of Technology";
        },
        "the module copyright string");

    m.def(
        "license",
        []() -> std::string {
            return "\n"
                   "    altar 2.0\n"
                   "    Copyright (c) 2013-present ParaSim Inc.\n"
                   "    Copyright (c) 2010-present California Institute of Technology\n"
                   "    All Rights Reserved\n"
                   "\n"
                   "\n"
                   "    Redistribution and use in source and binary forms, with or without\n"
                   "    modification, are permitted provided that the following conditions\n"
                   "    are met:\n"
                   "\n"
                   "    * Redistributions of source code must retain the above copyright\n"
                   "      notice, this list of conditions and the following disclaimer.\n"
                   "\n"
                   "    * Redistributions in binary form must reproduce the above copyright\n"
                   "      notice, this list of conditions and the following disclaimer in\n"
                   "      the documentation and/or other materials provided with the\n"
                   "      distribution.\n"
                   "\n"
                   "    * Neither the name \"altar\" nor the names of its contributors may be\n"
                   "      used to endorse or promote products derived from this software\n"
                   "      without specific prior written permission.\n"
                   "\n"
                   "    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS\n"
                   "    \"AS IS\" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT\n"
                   "    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS\n"
                   "    FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE\n"
                   "    COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,\n"
                   "    INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,\n"
                   "    BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;\n"
                   "    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER\n"
                   "    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT\n"
                   "    LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN\n"
                   "    ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE\n"
                   "    POSSIBILITY OF SUCH DAMAGE.\n";
        },
        "the module license string");

    m.def(
        "version",
        []() -> std::string { return "2.0"; },
        "the module version string");

    return;
}

// end of file
