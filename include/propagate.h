#ifndef QUARTZ_PROPAGATE_H
#define QUARTZ_PROPAGATE_H

#include "printer.h"

namespace quartz {

template<typename State, typename Potential>
using Propagator = std::function<
    State(const State &,
          const Potential &,
          const double)>;

template<typename State, typename Potential>
Propagator<State, Potential> operator+
    (const Propagator<State, Potential> & A,
     const Propagator<State, Potential> & B) {
  return [A, B](const State & old_state,
                const Potential & potential,
                const double dt) -> State {

    return A(old_state, potential, dt) + B(old_state, potential, dt);
  };
}


//TODO(Rui): Check if printer can be constant
// when the printer is going to change the value of,
// for example, json tree

//TODO(Rui): Check if the propagation method such as
// Runge-kutta method can be integrated in a neat way
template<typename State,
         typename Potential,
         typename Output>
State
propagate(const State & state,
          const Propagator<State, Potential> & propagator,
          const Potential & potential,
          const printer::Printer<Output, State> & printer,
          const double dt,
          const int print_level = 1) {
  //TODO(Rui): Stuff this function
}

}

#endif //QUARTZ_PROPAGATE_H
