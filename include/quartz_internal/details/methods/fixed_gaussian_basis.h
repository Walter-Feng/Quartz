#ifndef QUARTZ_FIXED_GAUSSIAN_BASIS_H
#define QUARTZ_FIXED_GAUSSIAN_BASIS_H

#include "quartz_internal/util/auto_generator.h"

namespace method {
// use your method name to create a subspace for your
// implementation of details
namespace fgb {

namespace details {

inline
arma::mat overlap_matrix(const arma::mat & points,
                         const arma::cube & covariances) {

  if (points.n_cols != covariances.n_slices) {
    throw Error(
        "Different number of points between the points and covariances");
  }

  if (points.n_rows != covariances.n_rows) {
    throw Error("Different dimension between the points and covariances");
  }

  arma::mat overlap(points.n_cols, points.n_cols);

#pragma omp parallel for
  for (arma::uword i = 0; i < overlap.n_rows; i++) {
    for (arma::uword j = i; j < overlap.n_cols; j++) {

      const arma::vec col_i = points.col(i);
      const arma::vec col_j = points.col(j);

      const auto gaussian =
          math::Gaussian<double>(covariances.slice(i), col_i) *
          math::Gaussian<double>(covariances.slice(j), col_j);

      overlap(i, j) = gaussian.integral();

    }
  }

  overlap = overlap + overlap.t();
  overlap.diag() /= 2.0;

  return overlap;
}


template<typename Function>
auto linear_combination_in_gaussian_basis(const Function & function,
                                          const arma::mat & overlap,
                                          const arma::mat & points,
                                          const arma::cube & covariances) {

  if (points.n_cols != covariances.n_slices) {
    throw Error(
        "Different number of points between the points and covariances");
  }

  if (points.n_rows != covariances.n_rows) {
    throw Error("Different dimension between the points and covariances");
  }

  if (overlap.n_rows != points.n_cols) {
    throw Error("Mismatch between the overlap matrix and points");
  }

  const arma::vec test_gaussian_mean = points.col(0);
  const arma::mat test_gaussian_covariance = covariances.slice(0);
  const auto test_gaussian = math::Gaussian<double>(test_gaussian_covariance,
                                                    test_gaussian_mean);

  const auto test_multiplied = function * test_gaussian;
  auto integral = test_multiplied.integral();
  arma::Col<decltype(integral)> expectations =
      arma::Col<decltype(integral)>(points.n_cols).eval();

#pragma omp parallel for
  for (arma::uword i = 0; i < expectations.n_elem; i++) {
    const arma::vec col_i = points.col(i);
    const auto multiplied =
        function * math::Gaussian<double>(covariances.slice(i),
                                          col_i);

    expectations(i) = multiplied.integral();
  }

  const arma::Col<decltype(integral)> coefs = arma::inv(overlap) * expectations;
  return coefs;
}

inline
arma::cube covariances_generator(const arma::uvec & grid,
                                 const arma::mat & range,
                                 const double widening = 1.0) {

  if (grid.n_elem != range.n_rows) {
    throw Error("Different dimension between the grid and range");
  }

  const arma::vec spacing =
      (range.col(1) - range.col(0)) / (grid - 1) * widening;

  arma::cube result(spacing.n_elem, spacing.n_elem, arma::prod(grid));

#pragma omp parallel for
  for (arma::uword i = 0; i < result.n_slices; i++) {
    result.slice(i) = arma::diagmat(spacing);
  }

  return result;
}

} // namespace details

struct State {
public:
  arma::mat points;
  arma::cube covariances;
  arma::mat overlap;
  arma::vec weights;
  arma::vec masses;

  // Establish an easy way to construct your State
  template<typename PhaseSpaceDistribution>
  State(const PhaseSpaceDistribution & initial,
        const arma::uvec & grid,
        const arma::mat & range,
        const arma::vec & masses,
        const double widening = 1.0) :
      points(math::space::points_generate(grid, range)),
      covariances(details::covariances_generator(grid, range, widening)),
      overlap(details::overlap_matrix(points, covariances)),
      weights(details::linear_combination_in_gaussian_basis(initial, overlap,
                                                            points,
                                                            covariances)),
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
        const arma::mat & range,
        const double widening = 1.0) :
      points(math::space::points_generate(grid, range)),
      covariances(details::covariances_generator(grid, range, widening)),
      overlap(details::overlap_matrix(points, covariances)),
      weights(details::linear_combination_in_gaussian_basis(initial, overlap,
                                                            points,
                                                            covariances)),
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
        const arma::mat & overlap,
        const arma::cube & covariances,
        const arma::vec & masses) :
      points(points),
      covariances(covariances),
      overlap(overlap),
      weights(weights),
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
  State normalise() const {
    return State(this->points,
                 arma::normalise(this->weights),
                 this->overlap,
                 this->covariances,
                 this->masses);
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

    math::Gaussian<double> result(this->covariances.slice(i),
                                  this->points.col(i));

    return result;
  }

  State operator+(const State & B) const {
    if (!arma::approx_equal(this->points, B.points, "abs_diff", 1e-16) ||
        !arma::approx_equal(this->masses, B.masses, "abs_diff", 1e-16) ||
        !arma::approx_equal(this->covariances, B.covariances, "abs_diff",
                            1e-16)) {
      throw Error("Different fgb states are being added");
    }

    return State(this->points, this->weights + B.weights, this->overlap,
                 this->covariances, this->masses);
  }

  State operator*(const double B) const {
    return State(this->points, this->weights * B, this->overlap,
                 this->covariances,
                 this->masses);
  }
};

struct Operator {

public:
  math::Polynomial<double> hamiltonian;
  arma::mat fock;
  arma::mat propagation_matrix;


  Operator(const State & state,
           const math::Polynomial<double> & potential) :
      hamiltonian(quartz::hamiltonian(potential)) {

    const arma::uword total = state.covariances.n_slices;
    arma::mat f(total, total, arma::fill::zeros);

#pragma omp parallel for
    for (arma::uword i = 0; i < total; i++) {
      for (arma::uword j = i+1; j < total; j++) {
        const auto gaussian_i = math::GaussianWithPoly(state.packet(i));
        const auto gaussian_j = math::GaussianWithPoly(state.packet(j));

        const auto post_functor =
            [&gaussian_j](const math::GaussianWithPoly<double> & b) -> double {
              const auto multiplied = gaussian_j * b;
              return multiplied.integral();
            };

        f(i, j) = moyal_bracket(post_functor,
                                gaussian_i,
                                this->hamiltonian,
                                this->hamiltonian.grade());
      }
    }

    this->fock = f - f.t();
    this->propagation_matrix = arma::inv(state.overlap) * this->fock;

  }


  inline
  PropagationType propagation_type() const {
    return Classic;
  }

  State operator()(const State & state) const {
    return State(state.points, this->propagation_matrix * state.weights,
                 state.overlap, state.covariances, state.masses);
  }
};
} // namespace fgb
}

#endif //QUARTZ_FIXED_GAUSSIAN_BASIS_H
