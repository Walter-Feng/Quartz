#ifndef METHODS_PACKET_H
#define METHODS_PACKET_H

#include "quartz_internal/details/math/gaussian.h"
#include "quartz_internal/details/math/space.h"
#include "quartz_internal/details/math/moyal_bracket.h"
#include "quartz_internal/util/auto_generator.h"

namespace method {
// use your method name to create a subspace for your
// implementation of details
namespace packet {

namespace details {

inline
arma::cube covariances_generator(const arma::mat & covariance,
                                 const arma::vec & mean,
                                 const arma::mat & points,
                                 const arma::vec & weights) {

  if (points.n_cols != weights.n_elem) {
    throw Error("the number of points does not match the number of weights");
  }

  if (!covariance.is_symmetric()) {
    throw Error("the covariance matrix provided is not symmetric");
  }

  arma::cube result(covariance.n_rows, covariance.n_cols, weights.n_elem);

  arma::mat cov = covariance + mean * mean.t() -
                  points * arma::diagmat(weights) * points.t() /
                  arma::sum(weights);

#pragma omp parallel for
  for (arma::uword i = 0; i < weights.n_elem; i++) {
    result.slice(i) = (cov + cov.t()) / 2.0;
  }

  return result;
}

} // namespace details

struct State {
public:
  arma::mat points;
  arma::vec weights;
  arma::cube covariances;
  arma::vec masses;

  // Establish an easy way to construct your State
  template<typename PhaseSpaceDistribution>
  State(const PhaseSpaceDistribution & initial,
        const arma::uvec & grid,
        const arma::mat & range,
        const arma::vec & masses) :
      points(math::space::points_generate(grid, range)),
      weights(arma::real(at(initial, points))),
      covariances(
          details::covariances_generator(initial.cov(), initial.mean(),
                                         points, weights)),
      masses(masses) {
    if (grid.n_elem % 2 != 0) {
      throw Error("Odd number of dimension - it is not likely a phase space");
    }
    if (grid.n_rows != range.n_rows) {
      throw Error("Different dimension between the grid and the range");
    }
    if (grid.n_rows != 2 * masses.n_rows) {
      throw Error("Different dimension between the grid and the masses");
    }
  }

  template<typename PhaseSpaceDistribution>
  State(const PhaseSpaceDistribution & initial,
        const arma::uvec & grid,
        const arma::mat & range) :
      points(math::space::points_generate(grid, range)),
      weights(arma::real(at(initial, points))),
      covariances(
          details::covariances_generator(initial.cov(), initial.mean(),
                                         points, weights)),
      masses(arma::ones<arma::vec>(grid.n_elem / 2)) {
    if (grid.n_elem % 2 != 0) {
      throw Error("Odd number of dimension - it is not likely a phase space");
    }
    if (grid.n_rows != range.n_rows) {
      throw Error("Different dimension between the grid and the range");
    }
    if (grid.n_rows != 2 * masses.n_rows) {
      throw Error("Different dimension between the grid and the masses");
    }
  }

  inline
  State(const arma::mat & points,
        const arma::vec & weights,
        const arma::cube & covariances,
        const arma::vec & masses) :
      points(points),
      weights(weights),
      covariances(covariances),
      masses(masses) {
    if (points.n_cols != weights.n_elem) {
      throw Error("Different number of points and corresponding weights");
    }

    if (points.n_rows != 2 * masses.n_rows) {
      throw Error("Different dimension between the points and the masses");
    }
  }

  inline
  arma::uword dim() const {
    return points.n_rows / 2;
  }

  inline
  arma::vec mean() const {
    return this->points * this->weights / arma::sum(this->weights);
  }

  inline
  arma::vec positional_expectation() const {
    arma::uword dim = this->dim();

    return this->points.rows(0, dim - 1) * this->weights /
           arma::sum(this->weights);
  }

  inline
  arma::vec momentum_expectation() const {
    arma::uword dim = this->dim();

    return this->points.rows(dim, 2 * dim - 1) * this->weights /
           arma::sum(this->weights);
  }

  inline
  arma::mat covariance_expectation() const {
    arma::mat result(arma::size(this->covariances.slice(0)), arma::fill::zeros);

#pragma omp parallel for
    for (arma::uword i = 0; i < this->covariances.n_slices; i++) {
      result += this->covariances.slice(i) * weights(i);
    }
    return (result +
            this->points * arma::diagmat(this->weights) * this->points.t()) /
           arma::sum(weights) - this->mean() * this->mean().t();
  }

  inline
  math::Gaussian<double> packet(const arma::uword i) const {

    if (this->covariances.n_slices <= i) {
      throw Error("packet enquiry out of bound");
    }

    math::Gaussian<double> result(this->covariances.slice(i),this->points.col(i));

    return result;
  }

  State operator+(const State & B) const {
    if (!arma::approx_equal(this->weights, B.weights, "abs_diff", 1e-16) ||
        !arma::approx_equal(this->masses, B.masses, "abs_diff", 1e-16)) {
      throw Error("Different md states are being added");
    }

    return State(this->points + B.points, this->weights,
                 this->covariances + B.covariances, this->masses);
  }

  State operator*(const double B) const {
    return State(this->points * B, this->weights, this->covariances * B, this->masses);
  }
};

struct Operator {

public:
  math::Polynomial<double> potential;


  Operator(const State & state,
           const math::Polynomial<double> & potential) :
      potential(potential) {}


  inline
  PropagationType propagation_type() const {
    return Classic;
  }

  State operator()(const State & state) const {

    // d/dt <x> = <p> / m
    const arma::mat position_change =
        arma::diagmat(1.0 / state.masses) *
        state.points.rows(state.dim(), 2 * state.dim() - 1);

    arma::mat momentum_change(state.dim(), state.covariances.n_slices);

    // d/dt <p> = - <d/dx V>
#pragma omp parallel for
    for (arma::uword i = 0; i < state.dim(); i++) {
      auto p_variable = math::Polynomial<double>(2 * state.dim(), 1.0);
      p_variable.indices(state.dim() + i, 0) = 1;
      const auto p_moyal_bracket =
          math::moyal_bracket(p_variable, hamiltonian(potential), p_variable.grade());
      for (arma::uword j = 0; j < momentum_change.n_cols; j++) {
        momentum_change(i,j) =
            state.packet(j).expectation(p_moyal_bracket);
      }
    }

    const arma::mat point_change = arma::join_cols(position_change,momentum_change);

    arma::cube covariances_change(arma::size(state.covariances));
    // d/dt <xi xj>
#pragma omp parallel for
    for(arma::uword i=0; i<state.covariances.n_rows; i++) {
      for(arma::uword j=0; j<state.covariances.n_cols; j++) {
        auto variable = math::Polynomial<double>(2*state.dim(), 1.0);
        variable.indices(i,0) = 1;
        variable.indices(j,0) = 1;
        const auto variable_moyal_bracket =
            math::moyal_bracket(variable,
                hamiltonian(this->potential, state.masses), variable.grade());
        for(arma::uword k=0; k<state.covariances.n_slices; k++) {
          covariances_change(i,j,k) = state.packet(k).expectation(variable_moyal_bracket);
        }
      }
    }

    return State(point_change, state.weights, covariances_change, state.masses);
  }

};

} // namespace md
}


#endif //METHODS_PACKET_H
