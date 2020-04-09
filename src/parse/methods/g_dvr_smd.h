#ifndef PARSE_METHODS_G_CWA_SMD_H
#define PARSE_METHODS_G_CWA_SMD_H

#include <quartz>
#include <boost/property_tree/ptree.hpp>

#include "src/util/ptree.h"
#include "src/parse/printer.h"

namespace quartz {

namespace ptree = boost::property_tree;

template<typename Initial>
ptree::ptree g_cwa_smd(const ptree::ptree & input,
                       const math::Polynomial<double> & potential,
                       const Initial & initial) {

  const arma::uvec grid =
      arma::uvec(util::get_list<arma::uword>(input.get_child("grid")));

  const arma::mat range =
      util::get_mat<double>(input.get_child("range")).t();

  arma::vec masses = arma::ones(grid.n_elem / 2);

  const auto steps = input.get<arma::uword>("steps");
  const auto dt = input.get<double>("dt");
  const arma::uword grade = input.get<double>("grade", 4) + 1;

  if (input.get_child_optional("mass")) {
    masses = arma::vec(util::get_list<double>(input.get_child("mass")));
  }

  method::g_cwa_smd::State initial_state(initial.wigner_transform(), grid, range,
                                       masses, grade);
  method::g_cwa_smd::Operator op(initial_state, potential);
  auto wrapper =
      math::runge_kutta_4<
          method::g_cwa_smd::Operator,
          method::g_cwa_smd::State,
          math::Polynomial<double>>;

  ptree::ptree result;

  auto printer_pair = printer(input, result, initial_state);

  propagate(initial_state, op, wrapper, potential, printer_pair.first, steps,
            dt, printer_pair.second);

  return result;
}

}

#endif