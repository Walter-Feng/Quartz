#ifndef PARSE_METHODS_DVR_SMD_H
#define PARSE_METHODS_DVR_SMD_H

#include <quartz>
#include <boost/property_tree/ptree.hpp>

#include "src/util/ptree.h"
#include "src/parse/printer.h"

namespace quartz {

namespace ptree = boost::property_tree;

template<typename Initial>
ptree::ptree dvr_smd(const ptree::ptree & input,
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

  method::dvr_smd::State initial_state(initial, grid, range, masses, grade);
  method::dvr_smd::Operator op(initial_state, potential);
  auto wrapper =
      method::dvr_smd::mixed_runge_kutta_4<
          method::dvr_smd::Operator,
          method::dvr_smd::State,
          math::Polynomial<double>>;

  ptree::ptree result;

  auto printer_pair = printer(input, result, initial_state);

  propagate(initial_state, op, wrapper, potential, printer_pair.first, steps,
            dt, printer_pair.second);

  return result;
}

template<typename Initial>
ptree::ptree dvr_smd(const ptree::ptree & input,
                     const MathObject<double> & potential,
                     const Initial & initial) {

  return dvr_smd(input, parse::polynomial(potential), initial);
}

}

#endif
