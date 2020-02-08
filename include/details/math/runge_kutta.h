#ifndef QUARTZ_RUNGE_KUTTA_H
#define QUARTZ_RUNGE_KUTTA_H

#include "propagate.h"
#include "error.h"
#include "util/check_member.h"

namespace math {

template<typename Operator, typename State, typename Potential>
Propagator<State, Potential>
runge_kutta_2(const Operator & liouville_operator,
              const Potential & potential) {

  static_assert(has_propagation_type<Operator, PropagationType(void)>::value,
                "Propagation type not specified");

  if (liouville_operator.propagation_type() == Schrotinger) {
    Warning(
        "Runge-Kutta method may not be suitable for Schrotinger's method");
  }

  if (has_time_evolve<Potential, void(const double &)>::value) {
    return [&liouville_operator](State state,
                                 const Potential & potential,
                                 const double dt) -> State {

      Potential potential_at_half_dt = potential;
      potential_at_half_dt.time_evolve(0.5 * dt);

      Potential potential_at_dt = potential;
      potential_at_dt.time_evolve(dt);

      const Operator operator_at_half_dt = Operator(state, potential_at_half_dt);

      const State k1 = dt * (liouville_operator * state);
      const State k2 = dt * (operator_at_half_dt * (0.5 * k1 + state));


      return state + k2;
    };
  }

  return [&liouville_operator](State state,
                               const Potential & potential,
                               const double dt) -> State {
    const State k1 = dt * (liouville_operator * state);
    const State k2 = dt * (liouville_operator * (0.5 * k1 + state));

    return state + k2;
  };
}

template<typename Operator, typename State, typename Potential>
Propagator<State, Potential>
runge_kutta_4(const Operator & liouville_operator,
              const Potential & potential) {

  static_assert(has_propagation_type<Operator, PropagationType(void)>::value,
                "Propagation type not specified");

  if (liouville_operator.propagation_type() == Schrotinger) {
    Warning(
        "Runge-Kutta method may not be suitable for Schrotinger's method");
  }

  if (has_time_evolve<Potential, void(const double &)>::value) {
    return [&liouville_operator](State state,
                                 const Potential & potential,
                                 const double dt) -> State {

      Potential potential_at_half_dt = potential;
      potential_at_half_dt.time_evolve(0.5 * dt);

      Potential potential_at_dt = potential;
      potential_at_dt.time_evolve(dt);

      const Operator operator_at_half_dt = Operator(state, potential_at_half_dt);
      const Operator operator_at_dt = Operator(state, potential_at_dt);

      const State k1 = dt * (liouville_operator * state);
      const State k2 = dt * (operator_at_half_dt * (0.5 * k1 + state));
      const State k3 = dt * (operator_at_half_dt * (0.5 * k2 + state));
      const State k4 = dt * (operator_at_dt * (state + k3));

      return state + (1.0 / 6.0) * k1 + (1.0 / 3.0) * k2 + (1.0 / 3.0) * k3 +
             (1.0 / 6.0) * k4;
    };
  }

  return [&liouville_operator](State state,
                               const Potential & potential,
                               const double dt) -> State {
    const State k1 = dt * (liouville_operator * state);
    const State k2 = dt * (liouville_operator * (0.5 * k1 + state));
    const State k3 = dt * (liouville_operator * (0.5 * k2 + state));
    const State k4 = dt * (liouville_operator * (state + k3));

    return state + (1.0 / 6.0) * k1 + (1.0 / 3.0) * k2 + (1.0 / 3.0) * k3 +
           (1.0 / 6.0) * k4;
  };
}


}

#endif //QUARTZ_RUNGE_KUTTA_H
