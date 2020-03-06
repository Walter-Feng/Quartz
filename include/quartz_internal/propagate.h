#ifndef QUARTZ_PROPAGATE_H
#define QUARTZ_PROPAGATE_H

#include "printer.h"

#include "quartz_internal/util/check_member.h"


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
     const Propagator<State> & B) {
  return [&A, &B](const State & old_state,
                  const double dt) -> State {

    return A(old_state, dt) + B(old_state, dt);
  };
}

template<typename Operator, typename State, typename Potential>
using OperatorWrapper = std::function<
    Propagator<State>(const Operator &, const Potential &)
>;

template<typename Operator,
    typename State,
    typename Potential>
OperatorWrapper<Operator, State, Potential>
operator<<(const OperatorWrapper<Operator, State, Potential> & A,
           const OperatorWrapper<Operator, State, Potential> & B) {

  //return in the form of OperatorWrapper
  return [A, B](const Operator & op, const Potential & potential)
      -> Propagator<State> {

    const Propagator<State> a = A(op, potential);
    const Propagator<State> b = B(op, potential);

    //OperatorWrapper requires returning of Propagator
    return [a, b](const State state,
                    const double dt) -> State {
      const State intermediate_state = a(state, dt);
      return b(intermediate_state, dt);
    };
  };
}

//TODO(Rui): Check if printer can be constant
// when the printer is going to change the value of,
// for example, json tree

template<typename State,
    typename Potential,
    typename Operator>
State
propagate(const State & initial_state,
          const Operator & op,
          const OperatorWrapper<Operator, State, Potential> & operator_wrapper,
          const Potential & potential,
          const Printer<State> & printer,
          const arma::uword steps,
          const double dt,
          const int print_level = 1) {

  std::cout << "Library: Quartz" << std::endl;
  std::cout << "version: " + version << std::endl << std::endl;

  //time dependent version, need to constantly update the propagator
  if constexpr(has_time_evolve<Potential, void(const double &)>::value) {

    Potential updated_potential = potential;

    //print out initial state & header

    printer(initial_state, 0, 0.0, print_level, true);

    State state = initial_state;

    for (arma::uword i = 1; i <= steps; i++) {
      state = propagator(state, dt);
      printer(state, i, i * dt, print_level, false);
      updated_potential.time_evolve(dt);
    }

    std::cout << std::endl;
    std::cout << "Quartz terminated normally." << std::endl;

    return state;
  }

    // time independent, directly generate propagator
  else {
    //print out initial state & header

    printer(initial_state, 0, 0.0, print_level, true);

    State state = initial_state;
    const Propagator<State> propagator =
        operator_wrapper(op, potential);

    for (arma::uword i = 1; i <= steps; i++) {
      state = std::move(propagator(state, dt));
      printer(state, i, i * dt, print_level, false);
    }

    std::cout << std::endl;
    std::cout << "Quartz terminated normally." << std::endl;

    return state;
  }

}


#endif //QUARTZ_PROPAGATE_H
