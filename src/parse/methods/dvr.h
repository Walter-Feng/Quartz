#ifndef PARSE_METHODS_DVR_H
#define PARSE_METHODS_DVR_H

#include <quartz>
#include <boost/property_tree/ptree.hpp>

#include "src/util/ptree.h"
#include "src/parse/printer.h"
#include "src/util/json_printer.h"

namespace quartz {

namespace dvr_details {

template<typename DVRState>
PtreePrinter<DVRState>
    state_printer = [](ptree::ptree & result_tree)
    -> Printer<DVRState> {
  return [&result_tree](const DVRState & state,
                        const arma::uword &,
                        const double,
                        const int,
                        const bool print_header) -> int {

    if(print_header) {
      util::put(result_tree, "grid", state.points);
    }
    util::put(result_tree, "coefs", state.coefs);

    return 0;
  };
};

}
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

  if(input.get<bool>("printer.print_state", false)) {
    printer_pair =
        printer(input,
                result,
                initial_state,
                {dvr_details::state_printer<method::dvr::State>},
                "state");
  }

  propagate(initial_state, op, wrapper, potential, printer_pair.first, steps,
            dt, printer_pair.second);

  return result;
}

}

#endif