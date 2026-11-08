// QS-only native backend. Kernel provenance: PROVENANCE.json; GPL-3.0-or-later.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
namespace py = pybind11;
template <class T> using arr = py::array_t<T, py::array::c_style | py::array::forcecast>;
#include "static_kernels.hpp"
#include "hmatrix_kernels.hpp"
#include "compressed_kernels.hpp"
PYBIND11_MODULE(_core, m) {
    m.def("static_pairs", &static_pairs);
    m.def("static_fill", &static_fill);
    m.def("static_combine", &static_combine);
    m.def("static_thread_count", &static_thread_count);
    m.def("hmatrix_apply", &hmatrix_apply);
    m.def("static_block", &static_block);
    m.def("rwg_inductance", &rwg_inductance);
#ifdef _OPENMP
    m.attr("static_openmp") = true;
#else
    m.attr("static_openmp") = false;
#endif
}
