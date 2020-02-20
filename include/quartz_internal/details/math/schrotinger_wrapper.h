#ifndef QUARTZ_SCHROTINGER_WRAPPER_H
#define QUARTZ_SCHROTINGER_WRAPPER_H

#include "util/check_member.h"

namespace math {

template<typename Operator, typename State, typename Potential>
const OperatorWrapper<Operator, State, Potential>
    schrotinger_wrapper = [](const Operator & operator_matrix,
                             const Potential & potential) -> Propagator<State> {
  static_assert(has_propagation_type<Operator, PropagationType(void)>::value,
                "Propagation type not specified");

  static_assert(has_inv<Operator, Operator(void)>::value,
                "Inverse of the operator not specified");

  if (operator_matrix.propagation_type() != Schrotinger) {
    Error(
        "Schrotinger wrapper is only suitable for Schrotinger's method");
  }

  if constexpr(has_time_evolve < Potential, void(const double &)>::value) {
  return [&operator_matrix, &potential](const State & state,
                                        const double dt) -> State {

    const arma::cx_mat unit_matrix = arma::eye<arma::cx_mat>
        (arma::size(operator_matrix.hamiltonian));

    Potential potential_at_half_dt = potential;
    potential_at_half_dt.time_evolve(0.5 * dt);

    const Operator operator_at_half_dt = Operator(state,
                                                  potential_at_half_dt);

    const Operator lhs =
        (Operator(arma::eye(arma::size(operator_matrix.hamiltonian)))
        + 0.5 * dt * cx_double{0.0, 1.0})(operator_at_half_dt);

    const Operator rhs =
        (Operator(arma::eye(arma::size(operator_matrix.hamiltonian)))
        - 0.5 * dt * cx_double{0.0, 1.0})(operator_at_half_dt);

    return (lhs.inv() * rhs)(state);
  };
}
  else {
    return [&operator_matrix](const State & state,
                              const double dt) -> State {

      const arma::cx_mat unit_matrix = arma::eye<arma::cx_mat>
          (arma::size(operator_matrix.hamiltonian));

      const Operator lhs =
          Operator(unit_matrix) +
          operator_matrix * (0.5 * dt * cx_double{0.0, 1.0});

      const Operator rhs =
          Operator(unit_matrix) -
          operator_matrix * (0.5 * dt * cx_double{0.0, 1.0});

      return (lhs.inv() * rhs)(state);
    };
  }
};

}

#endif //QUARTZ_SCHROTINGER_WRAPPER_H
