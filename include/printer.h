#ifndef QUARTZ_PRINTER_H
#define QUARTZ_PRINTER_H

namespace printer {
template<typename Output, typename State>
using Printer = std::function<Output(const State &, int)>;

template<typename State>
void generic_printer(const State & state,
                     const int print_level = 1) {

  std::ios_base::fmtflags f(std::cout.flags());

  const arma::vec real_space_expectation = state.positional_expectation();
  const arma::vec momentum_expectation = state.momentum_expectation();

  std::cout << "spacial  :" << real_space_expectation.t();
  std::cout << "momentum :" << momentum_expectation.t();

  std::cout.flags(f);

}

template<typename State>
void mute(const State & state,
          const int print_level = 0) {

}

}

#endif //QUARTZ_PRINTER_H
