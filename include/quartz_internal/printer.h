#ifndef QUARTZ_PRINTER_H
#define QUARTZ_PRINTER_H

#include "util/check_member.h"

template<typename T>
auto print(const arma::Mat<T> & arma,
           const int width = 10,
           const int precision = 6,
           const std::string aligned = ">") {
  for (arma::uword i = 0; i < arma.n_rows; i++) {
    for (arma::uword j = 0; j < arma.n_cols; j++) {
      const auto formatted =
          fmt::format("{:.{}}", arma(i, j), precision);
      fmt::print("{:" + aligned + "{}}", formatted, width);
    }
    fmt::print("\n");
  }
}

template<typename T>
auto print(const T number,
           const int width = 10,
           const int precision = 6,
           const std::string aligned = ">") {
  const auto formatted =
      fmt::format("{:.{}}", number, precision);
  fmt::print("{:" + aligned + "{}}", formatted, width);
}


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

  if (print_level == 1) {
    const arma::vec real_space_expectation = state.positional_expectation();
    if (print_header) {
      fmt::print("{:>{}}", "Step |", 6);
      fmt::print("{:>{}}", "Time |", width);
      fmt::print("{:>{}}", "Positional |",
                 width * real_space_expectation.n_elem);
      fmt::print("\n");
      for (arma::uword i = 0;
           i < 6 + width * (real_space_expectation.n_elem + 1); i++) {
        fmt::print("=");
      }
      fmt::print("\n");
    }

    fmt::print("{:>{}}", index, 6);

    const arma::mat combined =
        arma::join_rows(arma::vec{time}, real_space_expectation.t());
    print(combined, width, precision);
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
      fmt::print("Step |");
      fmt::print("{:>{}}", "Time |", width);
      fmt::print("{:>{}}", "Positional |",
                 width * real_space_expectation.n_elem);
      fmt::print("{:>{}}", "Positional |", width * momentum_expectation.n_elem);
      fmt::print("\n");
      for (arma::uword i = 0;
           i < 6 + width * (real_space_expectation.n_elem + 1) +
               width * momentum_expectation.n_elem; i++) {
        fmt::print("=");
      }
      fmt::print("\n");
    }

    fmt::print("{:>{}}", index, 6);

    const arma::mat combined =
        arma::join_rows(arma::vec{time},
                        arma::join_rows(real_space_expectation.t(),
                                        momentum_expectation.t()));

    print(combined, width, precision);
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
