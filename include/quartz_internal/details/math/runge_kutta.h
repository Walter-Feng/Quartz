#ifndef QUARTZ_RUNGE_KUTTA_H
#define QUARTZ_RUNGE_KUTTA_H

#include "quartz_internal/propagate.h"
#include "quartz_internal/error.h"
#include "quartz_internal/util/check_member.h"

namespace math {

template<typename Operator, typename State, typename Potential>
OperatorWrapper<Operator, State, Potential>
    runge_kutta_2 = [](const Operator & liouville_operator,
                       const Potential & potential) -> Propagator<State> {

  static_assert(has_propagation_type<Operator, PropagationType(void)>::value,
                "Propagation type not specified");

  if (liouville_operator.propagation_type() == Schrotinger) {
    Warning(
        "Runge-Kutta method may not be suitable for Schrotinger's method");
  }

  if constexpr(has_time_evolve<Potential, void(const double &)>::value) {
    return [&liouville_operator, &potential](const State & state,
                                             const double dt) -> State {

      Potential potential_at_half_dt = potential;
      potential_at_half_dt.time_evolve(0.5 * dt);

      Potential potential_at_dt = potential;
      potential_at_dt.time_evolve(dt);

      const Operator operator_at_half_dt = Operator(state,
                                                    potential_at_half_dt);

      const State k1 = liouville_operator(state) * dt;
      const State k2 = operator_at_half_dt(k1 * 0.5 + state) * dt;


      return state + k2;
    };
  }

  return [&liouville_operator, &potential](const State & state,
                                           const double dt) -> State {
    const State k1 = liouville_operator(state) * dt;
    const State k2 = liouville_operator(k1 * 0.5 + state) * dt;

    return state + k2;
  };
};

template<typename Operator, typename State, typename Potential>
OperatorWrapper<Operator, State, Potential>
    runge_kutta_4 = [](const Operator & liouville_operator,
                       const Potential & potential) -> Propagator<State> {

  static_assert(has_propagation_type<Operator, PropagationType(void)>::value,
                "Propagation type not specified");

  if (liouville_operator.propagation_type() == Schrotinger) {
    Warning(
        "Runge-Kutta method may not be suitable for Schrotinger's method");
  }

  if constexpr(has_time_evolve < Potential, void(const double &)>::value) {
  return [&liouville_operator, &potential](const State & state,
                                           const double dt) -> State {

    Potential potential_at_half_dt = potential;
    potential_at_half_dt.time_evolve(0.5 * dt);

    Potential potential_at_dt = potential;
    potential_at_dt.time_evolve(dt);

    const Operator operator_at_half_dt = Operator(state,
                                                  potential_at_half_dt);
    const Operator operator_at_dt = Operator(state, potential_at_dt);

    const State k1 = liouville_operator(state) * dt;
    const State k2 = operator_at_half_dt(k1 * 0.5 + state) * dt;
    const State k3 = operator_at_half_dt(k2 * 0.5 + state) * dt;
    const State k4 = operator_at_dt(state + k3) * dt;

    return state + k1 * (1.0 / 6.0) + k2 * (1.0 / 3.0) + k3 * (1.0 / 3.0) +
           k4 * (1.0 / 6.0);
  };
}

  return [&liouville_operator, &potential](const State & state,
                                           const double dt) -> State {
    const State k1 = liouville_operator(state) * dt;
    const State k2 = liouville_operator(k1 * 0.5 + state) * dt;
    const State k3 = liouville_operator(k1 * 0.5 + state) * dt;
    const State k4 = liouville_operator(state + k3) * dt;

    return state + k1 * (1.0 / 6.0) + k2 * (1.0 / 3.0) + k3 * (1.0 / 3.0) +
           k4 * (1.0 / 6.0);
  };
};


}

#endif //QUARTZ_RUNGE_KUTTA_H
