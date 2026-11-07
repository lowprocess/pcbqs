// QS-only native backend. Kernel provenance: PROVENANCE.json; GPL-3.0-or-later.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif
namespace py = pybind11;
template <class T> using arr = py::array_t<T, py::array::c_style | py::array::forcecast>;
#include "static_kernels.hpp"
PYBIND11_MODULE(_core, m) {
    m.def("static_pairs", &static_pairs);
    m.def("static_thread_count", &static_thread_count);
#ifdef _OPENMP
    m.attr("static_openmp") = true;
#else
    m.attr("static_openmp") = false;
#endif
}
