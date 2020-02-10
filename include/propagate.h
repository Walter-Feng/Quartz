#ifndef QUARTZ_PROPAGATE_H
#define QUARTZ_PROPAGATE_H

#include "printer.h"

#include "util/check_member.h"


enum PropagationType{
  Schrotinger,
  Classic
};

template<typename State, typename Potential>
using Propagator = std::function<
    State(State,
          const Potential &,
          const double)>;

template<typename State, typename Potential>
Propagator<State, Potential> operator+
    (const Propagator<State, Potential> & A,
     const Propagator<State, Potential> & B) {
  return [&A, &B](const State & old_state,
                const Potential & potential,
                const double dt) -> State {

    return A(old_state, potential, dt) + B(old_state, potential, dt);
  };
}

template<typename Operator, typename State, typename Potential>
using OperatorWrapper = std::function<
    Propagator<State, Potential>(const Operator &, const Potential &)
        >;


//TODO(Rui): Check if printer can be constant
// when the printer is going to change the value of,
// for example, json tree

template<typename State,
         typename Potential,
         typename Operator,
         typename Output>
State
propagate(const State & state,
          const Operator & op,
          const OperatorWrapper<Operator,State,Potential> & operator_wrapper,
          const Potential & potential, // to propagate the time-dependent potential
          const Printer<Output, State> & printer,
          const arma::uword steps,
          const double dt,
          const int print_level = 1) {

  //time dependent version, need to constantly update the propagator
  if(has_time_evolve<Potential, void(const double &)>::value) {
    Potential updated_potential = potential;



    for(arma::uword i=1;i<=steps;i++) {

    }
  }

}


#endif //QUARTZ_PROPAGATE_H
