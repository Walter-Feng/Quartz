#ifndef QUARTZ_PROPAGATE_H
#define QUARTZ_PROPAGATE_H

#include "printer.h"

namespace quartz {

template<typename State, typename Potential, typename T>
using Propagator = std::function<
    State(const State &,
          const Potential &,
          const double)>;


//TODO(Rui): Check if printer can be constant
// when the printer is going to change the value of,
// for example, json tree

//TODO(Rui): Check if the propagation method such as
// Runge-kutta method can be integrated in a neat way
template<typename State,
         typename Potential,
         typename Output,
         typename T>
State
propagate(const State & state,
          const Propagator<State, Potential, T> & propagator,
          const Potential & potential,
          const printer::Printer<Output, State> & printer,
          const double dt,
          const int print_level=1) {
  //TODO(Rui): Stuff this function
}

}

#endif //QUARTZ_PROPAGATE_H
