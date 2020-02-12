#ifndef QUARTZ_SCHROTINGER_WRAPPER_H
#define QUARTZ_SCHROTINGER_WRAPPER_H

#include "util/check_member.h"

namespace math {
namespace details {
template<typename T>
arma::cx_mat schrotinger_wrapper(const arma::Mat<T> & result, const double dt) {
  if (!result.is_square()) {
    throw Error("The matrix being wrapped for propagation is not square");
  }
  return
      arma::inv(arma::eye(result) + 0.5 * cx_double{0.0, 1.0} * dt * result) *
      arma::inv(arma::eye(result) - 0.5 * cx_double{0.0, 1.0} * dt * result);
}
}

template<typename Operator, typename State, typename Potential>
Propagator<State> schrotinger_wrapper(const Operator & operator_matrix,
                                      const Potential & potential) {
  static_assert(has_propagation_type<Operator, PropagationType(void)>::value,
                "Propagation type not specified");

  static_assert(has_inv<Operator, Operator(void)>::value,
                "Inverse of the operator not specified");

  if (operator_matrix.propagation_type() != Schrotinger) {
    Error(
        "Schrotinger wrapper is only suitable for Schrotinger's method");
  }

  if (has_time_evolve<Potential, void(const double &)>::value) {
    return [&operator_matrix, &potential](const State & state,
                                          const double dt) -> State {

      Potential potential_at_half_dt = potential;
      potential_at_half_dt.time_evolve(0.5 * dt);

      const Operator operator_at_half_dt = Operator(state,
                                                    potential_at_half_dt);

      const Operator lhs =
          arma::eye(arma::size(operator_matrix.hamiltonian))
          + 0.5 * dt * cx_double{0.0, 1.0} * operator_at_half_dt;

      const Operator rhs =
          arma::eye(arma::size(operator_matrix.hamiltonian))
          - 0.5 * dt * cx_double{0.0, 1.0} * operator_at_half_dt;

      return (lhs.inv() * rhs.inv()) * state;
    };
  }

  if (has_time_evolve<Potential, void(const double &)>::value) {
    return [&operator_matrix](State state,
                              const double dt) -> State {

      const Operator lhs =
          arma::eye(arma::size(operator_matrix.hamiltonian))
          + 0.5 * dt * cx_double{0.0, 1.0} * operator_matrix;

      const Operator rhs =
          arma::eye(arma::size(operator_matrix.hamiltonian))
          - 0.5 * dt * cx_double{0.0, 1.0} * operator_matrix;

      return (lhs.inv() * rhs.inv()) * state;
    };

  }

}

}

#endif //QUARTZ_SCHROTINGER_WRAPPER_H
