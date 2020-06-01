#ifndef PARSE_CWA_SMD_OPT_H
#define PARSE_CWA_SMD_OPT_H

#include <quartz>
#include <boost/property_tree/ptree.hpp>

#include "src/util/ptree.h"
#include "src/parse/printer.h"
#include "src/parse/math/polynomial.h"

namespace quartz {

namespace cwa_smd_opt_details {

Printer<method::cwa_smd_opt::State> state_printer(ptree::ptree & result_tree) {
  return [&result_tree](const method::cwa_smd_opt::State & state,
                        const arma::uword &,
                        const double,
                        const int,
                        const bool print_header) -> int {

    if(print_header) {
      util::put(result_tree, "weights", state.weights);
    }
    util::put(result_tree, "points", state.points);

    return 0;
  };
}

Printer<method::cwa_smd_opt::State> opt_printer(ptree::ptree & result_tree) {
  return [&result_tree](const method::cwa_smd_opt::State & state,
                        const arma::uword &,
                        const double,
                        const int,
                        const bool print_header) -> int {

    if(print_header) {
      util::put(result_tree, "weights", state.weights);
    }
    util::put(result_tree, "points", state.points);

    return 0;
  };
}
}


namespace ptree = boost::property_tree;

template<typename Initial>
ptree::ptree cwa_smd_opt(const ptree::ptree & input,
                         const math::Polynomial<double> & potential,
                         const Initial & initial) {

  const arma::uvec grid =
      arma::uvec(util::get_list<arma::uword>(input.get_child("grid")));

  const arma::mat range =
      util::get_mat<double>(input.get_child("range")).t();

  const arma::vec mean = arma::real(initial.mean());
  arma::vec scaling = arma::join_cols(mean, mean);
  if(input.get_child_optional("scaling")) {
    scaling = arma::vec(util::get_list<double>(input.get_child("scaling")));
  }

  const auto steps = input.get<arma::uword>("steps");
  const auto dt = input.get<double>("dt");
  const arma::uword grade = input.get<double>("grade", 4) + 1;

  arma::vec masses = arma::ones(grid.n_elem / 2);

  if (input.get_child_optional("mass")) {
    masses = arma::vec(util::get_list<double>(input.get_child("mass")));
  }

  const double tolerance = input.get<double>("tol", 0.1);
  const double initial_step_size = input.get<double>("init_step", 0.01);
  const double gradient_tolerance = input.get<double>("gradient_tol", 0.01);
  const arma::uword max_iter = input.get<arma::uword>("max_iter", 100);
  const std::string optimizer_type = input.get<std::string>("optimizer", "bfgs2");

  method::cwa_smd_opt::State initial_state(initial.wigner_transform(), grid,
                                           range, scaling,
                                           masses, grade);
  method::cwa_smd_opt::Operator op(initial_state, potential);
  auto wrapper =
      math::runge_kutta_4<
          method::cwa_smd_opt::Operator,
          method::cwa_smd_opt::State,
          math::Polynomial<double>>;

  ptree::ptree result;

  auto printer_pair = printer(input, result, initial_state);

  auto optimizer =
      method::cwa_smd_opt::cwa_opt<math::Polynomial<double>>(initial_step_size,
                                                             tolerance,
                                                             gradient_tolerance,
                                                             max_iter,
                                                             optimizer_type,
                                                             printer_pair.second);

  const auto propagator = wrapper << optimizer;

  propagate(initial_state, op, propagator, potential,
            printer_pair.first, steps,
            dt, printer_pair.second);

  return result;
}

template<typename Initial>
ptree::ptree cwa_smd_opt(const ptree::ptree & input,
                         const MathObject<double> & potential,
                         const Initial & initial) {

  return cwa_smd_opt(input, parse::polynomial(potential), initial);
}

}


#endif //PARSE_CWA_SMD_OPT_H
