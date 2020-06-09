#ifndef PARSE_METHODS_CWA_H
#define PARSE_METHODS_CWA_H

#include <quartz>
#include <boost/property_tree/ptree.hpp>

#include "src/util/ptree.h"
#include "src/parse/printer.h"

namespace quartz {

namespace cwa_details {

template<typename CWAState>
PtreePrinter<CWAState>
    state_printer = [](ptree::ptree & result_tree)
    -> Printer<CWAState> {
  return [&result_tree](const CWAState & state,
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
};

}

namespace ptree = boost::property_tree;

template<typename Potential, typename Initial>
ptree::ptree cwa(const ptree::ptree & input,
                 const Potential & potential,
                 const Initial & initial) {

  const arma::uvec grid =
      arma::uvec(util::get_list<arma::uword>(input.get_child("grid")));

  const arma::mat range =
      util::get_mat<double>(input.get_child("range")).t();

  arma::vec masses = arma::ones(grid.n_elem / 2);

  const auto steps = input.get<arma::uword>("steps");
  const auto dt = input.get<double>("dt");

  if (input.get_child_optional("mass")) {
    masses = arma::vec(util::get_list<double>(input.get_child("mass")));
  }

  method::cwa::State initial_state(initial.wigner_transform(), grid, range, masses);
  method::cwa::Operator<Potential> op(initial_state, potential);
  auto wrapper =
      math::runge_kutta_4<method::cwa::Operator<Potential>, method::cwa::State, Potential>;

  ptree::ptree result;

  auto printer_pair = printer(input, result, initial_state);

  propagate(initial_state, op, wrapper, potential, printer_pair.first, steps,
            dt, printer_pair.second);

  return result;
}

}

#endif //PARSE_METHODS_CWA_H
