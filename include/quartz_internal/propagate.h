#ifndef QUARTZ_PROPAGATE_H
#define QUARTZ_PROPAGATE_H

#include "printer.h"

#include "util/check_member.h"


enum PropagationType {
  Schrotinger,
  Classic
};

template<typename State>
using Propagator = std::function<
    State(const State &,
          const double)>;

template<typename State>
Propagator<State> operator+
    (const Propagator<State> & A,
     const Propagator<State,> & B) {
  return [&A, &B](const State & old_state,
                  const double dt) -> State {

    return A(old_state, dt) + B(old_state, dt);
  };
}

template<typename Operator, typename State, typename Potential>
using OperatorWrapper = std::function<
    Propagator<State>(const Operator &, const Potential &)
>;


//TODO(Rui): Check if printer can be constant
// when the printer is going to change the value of,
// for example, json tree

template<typename State,
    typename Potential,
    typename Operator,
    typename Output>
State
propagate(const State & initial_state,
          const Operator & op,
          const OperatorWrapper<Operator, State, Potential> & operator_wrapper,
          const Potential & potential,
          const Printer<Output, State> & printer,
          const arma::uword steps,
          const double dt,
          const int print_level = 1) {

  //time dependent version, need to constantly update the propagator
  if (has_time_evolve<Potential, void(const double &)>::value) {

    Potential updated_potential = potential;

    //print out initial state & header

    std::cout << "Library: Quartz" << std::endl;
    std::cout << "ver.   : -0.0.1" << std::endl;

    printer(initial_state, print_level, true);

    State state = initial_state;

    for (arma::uword i = 1; i <= steps; i++) {
      const Propagator<State> propagator =
          operator_wrapper(op, updated_potential);
      state = propagator(state, dt);
      printer(state, print_level);
      updated_potential.time_evolve(dt);
    }
  }

    // time independent, directly generate propagator
  else {
    //print out initial state & header

    std::cout << "Library: Quartz" << std::endl;
    std::cout << "ver.   : -0.0.1" << std::endl;

    printer(initial_state, print_level, true);

    State state = initial_state;
    const Propagator<State> propagator =
        operator_wrapper(op, potential);

    for(arma::uword i = 1; i <= steps; i++) {
      state = propagator(state, potential, dt);
      printer(state, print_level);
    }
  }

}


#endif //QUARTZ_PROPAGATE_H
