#ifndef QUARTZ_PROPAGATE_H
#define QUARTZ_PROPAGATE_H

namespace quartz {

template<typename T>
using Potential = std::function<double(const arma::Col<T> &)>;

template<typename State, typename T>
using Propagator = std::function<
    State(const State &,
          const Potential<T> &,
          const double)>;

template<typename Output, typename State>
using Printer = std::function<Output(const State &, int)>;


//TODO(Rui): Check if printer can be constant
// when the printer is going to change the value of,
// for example, json tree

//TODO(Rui): Check if the propagation method such as
// Runge-kutta method can be integrated in a neat way
template<typename State, typename Output, typename T>
State
propagate(const State & state,
          const Propagator<State, T> & propagator,
          const Potential<T> & potential,
          const Printer<Output, State> & printer,
          const double dt) {
  //TODO(Rui): Stuff this function
}

}

#endif //QUARTZ_PROPAGATE_H
