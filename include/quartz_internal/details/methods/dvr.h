// name your method
#ifndef METHOD_DVR_H
#define METHOD_DVR_H

// include only the necessary header files
#include "quartz_internal/propagate.h"
#include "details/math/polynomial.h"
#include "details/math/constants.h"
#include "details/math/space.h"

#include "util/member_function_wrapper.h"

namespace method {
// use your method name to create a subspace for your
// implementation of details
namespace dvr {

namespace details {

inline
cx_double kinetic_matrix_element(const long long i,
                                 const long long j,
                                 const double interval,
                                 const double mass) {
  if (i == j) {
    return cx_double{
        math::pi * math::pi / 6. / mass / interval / interval, 0.};
  } else {
    return cx_double{
        std::pow(-1, i - j) / mass / interval / interval / (double) (i - j) /
        (double) (i - j), 0.};
  }
}

inline
cx_double momentum_matrix_element(const long long i,
                                  const long long j,
                                  const double interval) {

  if (i == j) {
    return cx_double{0., 0.};
  } else {
    return -cx_double{0., std::pow(-1, i - j) / interval / (i - j)};
  }
}

inline
arma::cx_cube momentum_matrices(const arma::uvec & grid,
                                const arma::mat & ranges,
                                const arma::uword dim) {
  const long long narrowed_dim = arma::prod(grid);

  arma::cx_cube result = arma::cx_cube(narrowed_dim, narrowed_dim, dim);

  const arma::vec intervals = (ranges.col(1) - ranges.col(0))
                              / (grid - arma::ones(grid.n_elem));

  const auto table = math::space::grids_to_table(grid);

#pragma omp parallel for
  for (arma::uword k = 0; k < dim; k++) {
    for (long long i = 0; i < narrowed_dim; i++) {
      for (long long j = 0; j < narrowed_dim; j++) {
        result(i, j, k) = momentum_matrix_element(
            math::space::index_to_indices(i, table)[k],
            math::space::index_to_indices(j, table)[k],
            intervals(k));
      }
    }
  }

  return result;
}

inline
arma::cube position_matrices(const arma::mat & points) {
  arma::cube result = arma::cube(points.n_cols, points.n_cols, points.n_rows);
#pragma omp parallel for
  for (arma::uword i = 0; i < points.n_rows; i++) {
    result.slice(i) = arma::diagmat(points.row(i));
  }

  return result;
}

} // namespace details

struct State {
public:
  arma::mat points;
  arma::cx_vec coefs;
  arma::uvec grid;
  arma::mat ranges;
  arma::vec masses;
  arma::cube positional_matrices;
  arma::cx_cube momentum_matrices;

  // Establish an easy way to construct your State
  template<typename Wavefunction>
  State(const Wavefunction & initial,
        const arma::uvec & grid,
        const arma::mat & range,
        const arma::vec & masses) :
      points(math::space::points_generate(grid, range)),
      coefs(at(initial, points)),
      grid(grid),
      ranges(range),
      masses(masses),
      positional_matrices(dvr::details::position_matrices(points)),
      momentum_matrices(
          dvr::details::momentum_matrices(grid, range, range.n_rows)) {
    if (grid.n_rows != ranges.n_rows) {
      throw Error("Different dimension between the grid and the range");
    }
    if (grid.n_rows != masses.n_rows) {
      throw Error("Different dimension between the grid and the masses");
    }
  }

  template<typename Wavefunction>
  State(const Wavefunction & initial,
        const arma::uvec & grid,
        const arma::mat & range) :
      points(math::space::points_generate(grid, range)),
      coefs(at(initial, points)),
      grid(grid),
      ranges(range),
      masses(arma::ones<arma::vec>(arma::prod(grid))),
      positional_matrices(dvr::details::position_matrices(points)),
      momentum_matrices(
          dvr::details::momentum_matrices(grid, range, range.n_rows)) {
    if (grid.n_rows != ranges.n_rows) {
      throw Error("Different dimension between the grid and the range");
    }
  }

  inline
  State(const arma::cx_vec & coefs,
        const arma::uvec & grid,
        const arma::mat & range) :
      coefs(coefs),
      grid(grid),
      ranges(range),
      masses(arma::ones<arma::vec>(arma::prod(grid))),
      positional_matrices(dvr::details::position_matrices(points)),
      momentum_matrices(
          dvr::details::momentum_matrices(grid, range, range.n_rows)) {
    if (grid.n_rows != ranges.n_rows) {
      throw Error("Different dimension between the grid and the range");
    }
  }


  inline
  arma::cx_mat kinetic_energy_matrix() const {
    const long long narrowed_dim = arma::prod(this->grid);

    arma::cx_mat result = arma::zeros<arma::cx_mat>(narrowed_dim, narrowed_dim);

    const arma::vec intervals = (this->ranges.col(1) - this->ranges.col(0))
                                / (this->grid - arma::ones(this->grid.n_elem));


    const auto table = math::space::grids_to_table(this->grid);

#pragma omp parallel for
    for (long long i = 0; i < narrowed_dim; i++) {
      for (long long j = 0; j < narrowed_dim; j++) {
        for (arma::uword k = 0; k < this->grid.n_elem; k++) {

          result(i, j) += details::kinetic_matrix_element(
              math::space::index_to_indices(i, table)[k],
              math::space::index_to_indices(j, table)[k],
              intervals(k),
              this->masses(k));
        }
      }
    }

    return result;
  }

  template<typename Potential>
  arma::cx_mat hamiltonian_matrix(const Potential & potential) const {
    const arma::vec potential_diag = at(potential, this->points);

    this->kinetic_energy_matrix().print("kinetic energy");
    return this->kinetic_energy_matrix() + arma::diagmat(potential_diag);
  }


  inline
  arma::vec positional_expectation() const {
    arma::vec result = arma::vec(this->dim());
#pragma omp parallel for
    for (arma::uword i = 0; i < result.n_elem; i++) {
      const cx_double dimension_result =
          arma::cdot(this->coefs,
                    this->positional_matrices.slice(i) * this->coefs);
//      assert(std::abs(dimension_result.imag()) < 1e-8);
      result(i) = std::real(dimension_result);
    }

    return result / this->norm() / this->norm();
  }

  inline
  arma::vec momentum_expectation() const {
    arma::vec result = arma::vec(this->dim());
#pragma omp parallel for
    for (arma::uword i = 0; i < result.n_elem; i++) {
      const cx_double dimension_result =
          arma::cdot(this->coefs,
                    this->momentum_matrices.slice(i) * this->coefs);
      assert(std::abs(dimension_result.imag()) < 1e-8);
      result(i) = std::real(dimension_result);
    }

    return result / this->norm() / this->norm();
  }

  inline
  arma::uword dim() const {
    return this->grid.n_elem;
  }

  inline
  double norm() const {
    return arma::norm(this->coefs);
  }
};

struct Operator {

private:
  PropagationType type = Schrotinger;

public:
  arma::Mat<cx_double> hamiltonian;

  template<typename Potential>
  Operator(const State & state,
           const Potential & potential) :
      hamiltonian(state.hamiltonian_matrix(potential)) {}

  template<typename T>
  Operator(const arma::Mat<T> & operator_matrix) :
      hamiltonian(arma::conv_to<arma::cx_mat>::from(operator_matrix)) {}

  inline
  PropagationType propagation_type() const {
    return Schrotinger;
  }

  State operator*(State state) const {
    state.coefs = this->hamiltonian * state.coefs;
    return state;
  }

  Operator operator+(const Operator & B) const {
    const arma::cx_mat new_mat = this->hamiltonian + B.hamiltonian;
    return Operator(new_mat);
  }

  Operator operator-(const Operator & B) const {
    const arma::cx_mat new_mat = this->hamiltonian - B.hamiltonian;
    return Operator(new_mat);
  }

  Operator operator*(const Operator & B) const {
    const arma::cx_mat new_mat = this->hamiltonian * B.hamiltonian;
    return Operator(new_mat);
  }

  template<typename T>
  Operator operator*(const T & B) const {
    const arma::cx_mat new_mat = this->hamiltonian * B;
    return Operator(new_mat);
  }

  inline
  Operator inv() const {
    const arma::cx_mat inversed = arma::inv(this->hamiltonian);
    return Operator(inversed);
  }
};


} // namespace dvr
}

#endif //METHOD_DVR_H