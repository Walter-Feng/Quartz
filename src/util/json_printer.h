#ifndef QUARTZ_JSON_PRINTER_H
#define QUARTZ_JSON_PRINTER_H

#include <quartz>
#include <boost/property_tree/ptree.hpp>

#include "ptree.h"

namespace quartz {

namespace ptree = boost::property_tree;

template<typename State>
Printer<State>
ptree_printer(ptree::ptree & result_tree,
              const std::optional<Printer<State>> additional = std::nullopt,
              const std::string additional_label = "") {

  return [&result_tree, additional](const State & state,
                        const arma::uword & index,
                        const double time,
                        const int,
                        const bool) -> int {

    ptree::ptree step_result;

    static_assert(
        quartz::has_positional_expectation<State, arma::vec(void)>::value,
        "The state does not support exporting positional expectation values, "
        "therefore not support ptree printers");

    static_assert(
        quartz::has_momentum_expectation<State, arma::vec(void)>::value,
        "The state does not support exporting momentum expectation values, "
        "therefore not support ptree printers");

    util::put(step_result, "step", index);
    util::put(step_result, "time", time);
    util::put(step_result, "positional", state.positional_expectation());
    util::put(step_result, "momentum", state.positional_expectation());

    if(additional.has_value()) {
      ptree::ptree additional_ptree;

    }
    result_tree.push_back(std::make_pair("", step_result));

    return 0;
  };

}

template<typename State, typename Function>
Printer<State>
ptree_expectation_printer(ptree::ptree & result_tree,
                          const std::vector<Function> & observables) {

  return [&result_tree, observables](const State & state,
                                      const arma::uword,
                                      const double time,
                                      const int,
                                      const bool print_header = false) -> int {

    ptree::ptree step_result;

    static_assert(
        quartz::has_expectation<State, arma::vec(std::vector<Function>)>::value,
        "The state does not support exporting expectation values, "
        "therefore not support ptree printers");

    if (print_header) {
      std::vector<std::string> observables_string(observables.size());

      for (arma::uword i = 0; i < observables.size(); i++) {
        observables_string[i] = observables[i].to_string();
      }

      util::put<std::string>(step_result, "observables", observables_string);
    }

    step_result.put<double>("time", time);
    util::put(step_result, "time", time);
    util::put(step_result, "expectation", state.expectation(observables));

    result_tree.push_back(std::make_pair("", step_result));

    return 0;
  };

}


}

#endif //QUARTZ_JSON_PRINTER_H
