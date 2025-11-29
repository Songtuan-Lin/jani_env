#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/filesystem.h>
#include "engine.h"


namespace nb = nanobind;

NB_MODULE(backend, m) {
    nb::class_<JANIEngine>(m, "JANIEngine")
        .def(nb::init<const std::filesystem::path &,
                      const std::filesystem::path &,
                      const std::filesystem::path &,
                      const std::filesystem::path &,
                      const std::filesystem::path &,
                      int>(),
                    nb::arg("jani_model_path"),
                    nb::arg("jani_property_path"),
                    nb::arg("start_states_path"),
                    nb::arg("objective_path"),
                    nb::arg("failure_property_path"),
                    nb::arg("seed"))
        .def("get_num_actions", &JANIEngine::get_num_actions)
        .def("get_num_variables", &JANIEngine::get_num_variables)
        .def("get_num_constants", &JANIEngine::get_num_constants)
        .def("get_current_action_mask", &JANIEngine::get_current_action_mask)
        .def("test_guards_for_action", &JANIEngine::testGuardsForAction)
        .def("test_destinations_for_action", &JANIEngine::testDestinationsForAction)
        .def("reset", &JANIEngine::reset)
        .def("step", &JANIEngine::step);
}