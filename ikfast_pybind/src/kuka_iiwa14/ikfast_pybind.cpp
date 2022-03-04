#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
// #include <pybind11/eigen.h>
// #include <pybind11/iostream.h>

#ifndef IKFAST_HAS_LIBRARY
#define IKFAST_HAS_LIBRARY
#endif

#include <kuka_iiwa14/ikfast.h>

#ifdef IKFAST_NAMESPACE
using namespace IKFAST_NAMESPACE;
#endif

namespace py = pybind11;

PYBIND11_MODULE(ikfast_kuka_iiwa14, m)
{
    m.doc() = R"pbdoc(
        ikfast_kuka_iiwa14
        -----------------------
        .. currentmodule:: ikfast_kuka_iiwa14
        .. autosummary::
           :toctree: _generate
           get_ik
           get_fk
    )pbdoc";

    m.def("get_ik", [](const std::vector<double> &trans_list,
                       const std::vector<std::vector<double>> &rot_list)
    {
      using namespace ikfast;
      IkSolutionList<double> solutions;

      std::vector<double> vfree(GetNumFreeParameters());
      double eerot[9], eetrans[3];

      for(std::size_t i = 0; i < 3; ++i)
      {
          eetrans[i] = trans_list[i];
          std::vector<double> rot_vec = rot_list[i];
          for(std::size_t j = 0; j < 3; ++j)
          {
              eerot[3*i + j] = rot_vec[j];
          }
      }

      // call ikfast routine
      bool b_success = ComputeIk(eetrans, eerot, &vfree[0], solutions);

      std::vector<std::vector<double>> solution_list;
      if (!b_success)
      {
          //fprintf(stderr,"Failed to get ik solution\n");
          return solution_list; // Equivalent to returning None in python
      }

      std::vector<double> solvalues(GetNumJoints());

      // convert all ikfast solutions into a std::vector
      for(std::size_t i = 0; i < solutions.GetNumSolutions(); ++i)
      {
          const IkSolutionBase<double>& sol = solutions.GetSolution(i);
          std::vector<double> vsolfree(sol.GetFree().size());
          sol.GetSolution(&solvalues[0],
                          vsolfree.size() > 0 ? &vsolfree[0] : NULL);

          std::vector<double> individual_solution = std::vector<double>(GetNumJoints());
          for(std::size_t j = 0; j < solvalues.size(); ++j)
          {
              individual_solution[j] = solvalues[j];
          }
          solution_list.push_back(individual_solution);
      }
      return solution_list;
    },
    py::arg("trans_list"),
    py::arg("rot_list"),
    R"pbdoc(
        get inverse kinematic solutions for pybind_kawasaki_rs010n
    )pbdoc");

    m.def("get_fk", [](const std::vector<double> &joint_list)
    {
      using namespace ikfast;
      // eerot is a flattened 3x3 rotation matrix
      double eerot[9], eetrans[3];

      std::vector<double> joints(GetNumJoints());
      for(std::size_t i = 0; i < GetNumJoints(); ++i)
      {
          joints[i] = joint_list[i];
      }

      // call ikfast routine
      ComputeFk(&joints[0], eetrans, eerot);

      // convert computed EE pose to a python object
      std::vector<double> pos(3);
      std::vector<std::vector<double>> rot(3);

      for(std::size_t i = 0; i < 3; ++i)
      {
          pos[i] = eetrans[i];
          std::vector<double> rot_vec(3);
          for( std::size_t j = 0; j < 3; ++j)
          {
              rot_vec[j] = eerot[3*i + j];
          }
          rot[i] = rot_vec;
      }
      return std::make_tuple(pos, rot);
    },
    py::arg("joint_list"),
    R"pbdoc(
        get forward kinematic solutions for ikfast_kuka_iiwa14
    )pbdoc");

    m.def("get_dof", []()
    {
      return int(GetNumJoints());
    },
    R"pbdoc(
        get number dofs configured for the ikfast module
    )pbdoc");

    m.def("get_free_dof", []()
    {
      return int(GetNumFreeParameters());
    },
    R"pbdoc(
        get number of free dofs configured for the ikfast module
    )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif

}
