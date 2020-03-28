#ifndef QUARTZ_PRINTER_H
#define QUARTZ_PRINTER_H

#include "util/check_member.h"

template<typename T>
void print(const arma::Mat<T> & arma,
           const int width = 10,
           const int precision = 6,
           const std::string aligned = ">") {
  for (arma::uword i = 0; i < arma.n_rows; i++) {
    for (arma::uword j = 0; j < arma.n_cols; j++) {
      const std::string formatted =
          fmt::format("{:.{}}", arma(i, j), precision);
      fmt::print("{:" + aligned + "{}}", formatted, width);
    }
    fmt::print("\n");
  }
}

template<typename T>
void print(const arma::Row<T> & arma,
           const int width = 10,
           const int precision = 6,
           const std::string aligned = ">") {
  for (arma::uword j = 0; j < arma.n_cols; j++) {
    const std::string formatted =
        fmt::format("{:.{}}", arma(j), precision);
    fmt::print("{:" + aligned + "{}}", formatted, width);
  }
}

template<typename T>
void print(const T number,
           const int width = 10,
           const int precision = 6,
           const std::string aligned = ">") {
  const auto formatted =
      fmt::format("{:.{}}", number, precision);
  fmt::print("{:" + aligned + "{}}", formatted, width);
}

template<typename T>
std::string format(const arma::Mat<T> & arma,
                   const int width = 10,
                   const int precision = 6,
                   const std::string aligned = ">") {

  std::string result = "";

  for (arma::uword i = 0; i < arma.n_rows; i++) {
    for (arma::uword j = 0; j < arma.n_cols; j++) {
      const auto formatted =
          fmt::format("{:.{}}", arma(i, j), precision);
      result += fmt::format("{:" + aligned + "{}}", formatted, width);
    }
    result += fmt::format("\n");
  }

  return result;
}

template<typename T>
std::string format(const T number,
                   const int width = 10,
                   const int precision = 6,
                   const std::string aligned = ">") {
  const auto formatted =
      fmt::format("{:.{}}", number, precision);

  return fmt::format("{:" + aligned + "{}}", formatted, width);
}


template<typename State>
using Printer = std::function<int(const State & state,
                                  const arma::uword index,
                                  const double time,
                                  const int print_level,
                                  const bool print_header)>;

template<typename State>
Printer<State> generic_printer = [](const State & state,
                                    const arma::uword index,
                                    const double time,
                                    const int print_level = 1,
                                    const bool print_header = false) -> int {

  static_assert(has_positional_expectation<State, arma::vec(void)>::value,
                "The state does not support exporting positional expectation values, "
                "therefore not support generic printers");

  int width = 18;
  int precision = 8;

  if (print_level > 2) {
    width = 27;
    precision = 17;
  }

  int total_length = 0;

  if (print_level == 1) {
    const arma::vec real_space_expectation = state.positional_expectation();
    total_length = 6 + width * (real_space_expectation.n_elem + 1);

    if (print_header) {
      for (int i = 0; i < total_length; i++) {
        fmt::print("=");
      }
      fmt::print("\n");

      fmt::print("{:>{}}", "|Step|", 6);
      fmt::print("{:>{}}", "Time |", width);
      fmt::print("{:>{}}", "Positional |",
                 width * real_space_expectation.n_elem);
      fmt::print("\n");
      for (int i = 0; i < total_length; i++) {
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
    total_length = 6 + width * (real_space_expectation.n_elem + 1) +
                   width * momentum_expectation.n_elem;

    if (print_header) {
      for (int i = 0; i < total_length; i++) {
        fmt::print("=");
      }
      fmt::print("\n");
      fmt::print("|Step|");
      fmt::print("{:>{}}", "Time |", width);
      fmt::print("{:>{}}", "Positional |",
                 width * real_space_expectation.n_elem);
      fmt::print("{:>{}}", "Momentum |", width * momentum_expectation.n_elem);
      fmt::print("\n");
      for (int i = 0; i < total_length; i++) {
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

  return total_length;

};

template<typename State, typename Function>
Printer<State> expectation_printer(const std::vector<Function> & observables) {

  static_assert(has_expectation<State, arma::vec(std::vector<Function>)>::value,
                "The state does not support exporting arbitrary expectation values, "
                "therefore not support generic printers");

  return [observables](const State & state,
                        const arma::uword index,
                        const double time,
                        const int print_level = 1,
                        const bool print_header = false) -> int {

    std::vector<std::string> observables_string(observables.size());
    int max_width = 0;
    for (unsigned long i = 0; i < observables.size(); i++) {
      observables_string[i] = observables[i].to_string();
      max_width = std::max(max_width, (int) observables_string[i].length());
    }

    max_width += 3;

    int width = std::max(max_width, 18);
    int precision = width - 10;

    if (print_level > 2) {
      width = std::max(max_width, 27);
      precision = 17;
    }

    const int total_length = 6 + width * (observables.size() + 1);

    if (print_header) {
      for (int i = 0; i < total_length; i++) {
        fmt::print("=");
      }
      fmt::print("\n");

      fmt::print("{:>{}}", "|Step|", 6);
      fmt::print("{:>{}}", "Time |", width);
      fmt::print("{:>{}}", "Expectation |",
                 width * observables.size());
      fmt::print("\n");
      for (int i = 0; i < total_length; i++) {
        fmt::print("=");
      }
      fmt::print("\n");
      fmt::print("{:>{}}", "|", 6 + width);
      for (unsigned long i = 0; i < observables.size(); i++) {
        fmt::print("{:>{}}", observables[i].to_string() + " |", width);
      }
      fmt::print("\n");
      for (int i = 0; i < total_length; i++) {
        fmt::print("=");
      }
      fmt::print("\n");
    }

    fmt::print("{:>{}}", index, 6);
    print(time, width, precision);

    const arma::rowvec expectation = state.expectation(observables).t();

    print(expectation, width, precision);

    fmt::print("\n");

    return total_length;
  };
}

template<typename State>
Printer<State> mute = [](const State &,
                         const arma::uword,
                         const double,
                         const int,
                         const bool) -> int {

  return 0;
};

template<typename State>
Printer<State> operator<<(const Printer<State> a, const Printer<State> b) {
  return [a, b](const State & state,
                  const arma::uword index,
                  const double time,
                  const int print_level = 1,
                  const bool print_header = false) -> int {

    const int a_length = a(state, index, time, print_level, print_header);
    const int b_length = b(state, index, time, print_level, print_header);

    return std::max(a_length, b_length);

  };
}

#endif //QUARTZ_PRINTER_H
