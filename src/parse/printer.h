#ifndef PARSE_PRINTER_H
#define PARSE_PRINTER_H

#include <quartz>
#include <boost/property_tree/ptree.hpp>

#include "src/util/json_printer.h"

namespace quartz {

namespace ptree = boost::property_tree;


// Printer + print_level ( which is read from input )
template<typename State>
std::pair<Printer<State>, int> printer(const ptree::ptree & input,
                                       ptree::ptree & result,
                                       const State & state) {

  if (!input.get_child_optional("printer")) {
    return {generic_printer<State>, 1};
  }

  const ptree::ptree printer_input = input.get_child("printer");

  std::string type = printer_input.get("type", "generic");
  bool mute = printer_input.get("mute", false);
  int print_level = printer_input.get("print_level", 1);

  //print json
  if (input.get_optional<std::string>("json")) {
    if (type == "generic") {
      if (!mute)
        return
            {ptree_printer<State>(result) << generic_printer<State>,
             print_level};
      else return {ptree_printer<State>(result), print_level};
    } else if constexpr(has_expectation<State, arma::vec(
        std::vector<math::Polynomial<double>>)>::value) {
      if (type == "expectation") {

        arma::uword grade = printer_input.get("grade", 2);

        const std::vector<math::Polynomial<double>> op = restricted_polynomial_observables(
            2 * state.dim(), grade);

        if (!mute)
          return {ptree_expectation_printer<State>(result, op)
                      << expectation_printer<State>(op), print_level};
        else return {ptree_expectation_printer<State>(result, op), print_level};
      } else {
        throw Error("The printer required is not supported");
      }
    }
  } else {
    if (type == "generic") return {generic_printer<State>, print_level};
    else if constexpr(has_expectation<State, arma::vec(
        std::vector<math::Polynomial<double>>)>::value) {
      if (type == "expectation") {
        arma::uword grade = printer_input.get("grade", 2);
        const auto op = restricted_polynomial_observables(2 * state.dim(), grade);
        return {expectation_printer<State>(op), print_level};
      }
    } else throw Error("The printer required is not supported");
  }
}

}

#endif //PARSE_PRINTER_H
