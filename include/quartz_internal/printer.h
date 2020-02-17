#ifndef QUARTZ_PRINTER_H
#define QUARTZ_PRINTER_H

#include "util/check_member.h"

template<typename State>
using Printer = std::function<void(const State & state,
                                   const arma::uword index,
                                   const double time,
                                   const int print_level,
                                   const bool print_header)>;

template<typename State>
Printer<State> generic_printer = [](const State & state,
                                    const arma::uword index,
                                    const double time,
                                    const int print_level = 1,
                                    const bool print_header = false) -> void {

  static_assert(has_positional_expectation<State, arma::vec(void)>::value,
                "The state does not support exporting positional expectation values, "
                "therefore not support generic printers");

  int width = 18;
  int precision = 8;

  if (print_level > 2) {
    width = 27;
    precision = 17;
  }

  std::cout << std::setw(width) << std::setprecision(precision);

  if (print_level == 1) {
    const arma::vec real_space_expectation = state.positional_expectation();
    if (print_header) {
      std::cout << std::setw(width) << "Time |";
      std::cout << std::setw(width * real_space_expectation.n_elem)
                << "Positional |" << std::endl;
      for (arma::uword i = 0;
           i < width * (real_space_expectation.n_elem + 1); i++) {
        std::cout << "=";
      }
      std::cout << std::endl;

      std::cout << std::setw(width) << std::setprecision(precision);
    }

    arma::join_rows(arma::vec{time}, real_space_expectation.t()).raw_print(
        std::cout);
  }

  if (print_level >= 2) {
    if (!has_momentum_expectation<State, arma::vec(void)>::value) {
      throw Error(
          "The state does not support exporting momentum expectation values, "
          "therefore not support generic printers with print level higher than 0");
    }
    const arma::vec real_space_expectation = state.positional_expectation();
    const arma::vec momentum_expectation = state.momentum_expectation();

    if (print_header) {
      std::cout << std::setw(width) << "Time |";
      std::cout << std::setw(width * real_space_expectation.n_elem)
                << "Positional |";
      std::cout << std::setw(width * momentum_expectation.n_elem)
                << "Momentum |";
      std::cout << std::endl;
      for (arma::uword i = 0; i < width * (real_space_expectation.n_elem + 1) +
                                  width * momentum_expectation.n_elem; i++) {
        std::cout << "=";
      }
      std::cout << std::endl;
      std::cout << std::setw(width) << std::setprecision(precision);
    }

    arma::join_rows(arma::vec{time}, arma::join_rows(real_space_expectation.t(),
                                                     momentum_expectation.t())).raw_print(
        std::cout);
  }

};


template<typename State>
Printer<State> mute = [](const State &,
                         const arma::uword,
                         const double,
                         const int,
                         const bool) -> void {

};


#endif //QUARTZ_PRINTER_H
