#ifndef PARSE_METHODS_DVR_H
#define PARSE_METHODS_DVR_H

#include <quartz>
#include <boost/property_tree/ptree.hpp>

#include "src/util/ptree.h"
#include "src/parse/printer.h"

namespace quartz {

namespace ptree = boost::property_tree;

template<typename Potential, typename Initial>
ptree::ptree dvr(const ptree::ptree & input,
                 const Potential & potential,
                 const Initial & initial) {

  const arma::uvec grid =
      arma::uvec(util::get_list<arma::uword>(input.get_child("grid")));

  const arma::mat range =
      util::get_mat<double>(input.get_child("range")).t();

  arma::vec masses = arma::ones(grid.n_elem);

  const auto steps = input.get<arma::uword>("steps");
  const auto dt = input.get<double>("dt");

  if (input.get_child_optional("mass")) {
    masses = arma::vec(util::get_list<double>(input.get_child("mass")));
  }

  method::dvr::State initial_state(initial, grid, range, masses);
  method::dvr::Operator op(initial_state, potential);
  auto wrapper =
      math::schrotinger_wrapper<method::dvr::Operator, method::dvr::State, Potential>;

  ptree::ptree result;

  auto printer_pair = printer(input, result, initial_state);

  propagate(initial_state, op, wrapper, potential, printer_pair.first, steps,
            dt, printer_pair.second);

  return result;
}

}

#endif