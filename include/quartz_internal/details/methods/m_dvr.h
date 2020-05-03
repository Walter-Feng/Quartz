// name your method
#ifndef M_DVR_H
#define M_DVR_H

// include only the necessary header files
#include "cwa.h"
#include "quartz_internal/propagate.h"
#include "details/math/polynomial.h"
#include "details/math/constants.h"
#include "details/math/space.h"

#include "util/member_function_wrapper.h"


namespace method {
// use your method name to create a subspace for your
// implementation of details
namespace m_dvr {


namespace details {
//inline
//arma::cx_cube momentum_matrices(const arma::uvec & grid,
//                                const arma::mat & ranges,
//                                const arma::uword dim,
//                                const arma::uword layers) {
//  arma::cx_double
//}

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
  arma::uword layers;

  // Establish an easy way to construct your State
  template<typename Wavefunction>
  State(const std::vector<Wavefunction> & initial,
        const arma::uvec & grid,
        const arma::mat & range,
        const arma::vec & masses) :
      points(math::space::points_generate(grid, range)),
      coefs(),
      grid(grid),
      ranges(range),
      masses(masses),
      positional_matrices(dvr::details::position_matrices(points)),
      momentum_matrices(
          dvr::details::momentum_matrices(grid, range, range.n_rows)),
      layers(initial.size()) {
    if (grid.n_rows != ranges.n_rows) {
      throw Error("Different dimension between the grid and the range");
    }
    if (grid.n_rows != masses.n_rows) {
      throw Error("Different dimension between the grid and the masses");
    }

    for(auto i=0; i<initial.size(); i++) {
      this->coefs = arma::join_cols(this->coefs,
          arma::conv_to<arma::cx_vec>::from(at(initial[i], points)));
    }
  }

  template<typename Wavefunction>
  State(const std::vector<Wavefunction> & initial,
        const arma::uvec & grid,
        const arma::mat & range) :
      points(math::space::points_generate(grid, range)),
      coefs(),
      grid(grid),
      ranges(range),
      masses(arma::ones<arma::vec>(grid.n_elem)),
      positional_matrices(dvr::details::position_matrices(points)),
      momentum_matrices(
          dvr::details::momentum_matrices(grid, range, range.n_rows)),
      layers(initial.size()) {
    if (grid.n_rows != ranges.n_rows) {
      throw Error("Different dimension between the grid and the range");
    }

    for(auto i=0; i<initial.size(); i++) {
      this->coefs = arma::join_cols(this->coefs,
                                    arma::conv_to<arma::cx_vec>::from(at(initial[i], points)));
    }
  }

  inline
  State(const arma::cx_vec & coefs,
        const arma::uvec & grid,
        const arma::mat & range,
        const arma::vec & masses,
        const arma::uword layers) :
      coefs(coefs),
      grid(grid),
      ranges(range),
      masses(masses),
      positional_matrices(dvr::details::position_matrices(points)),
      momentum_matrices(
          dvr::details::momentum_matrices(grid, range, range.n_rows)),
      layers(layers) {
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

          result(i, j) += dvr::details::kinetic_matrix_element(
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
  arma::cx_mat hamiltonian_matrix(const arma::field<Potential> & potential) const {

    if(potential.n_cols != this->layers || potential.n_rows != this->layers) {
      throw Error("m_dvr: the layers of potential does not match the system");
    }

    arma::cx_mat result;

    for(arma::uword i=0; i<potential.n_rows; i++) {
      arma::cx_mat row_result;
      for(arma::uword j=0; j<potential.n_cols; j++) {

        const arma::cx_mat potential_diag = arma::diagmat(at(potential(i,j), this->points));

        if(i==j) {
          row_result = arma::join_rows(row_result,
                                       potential_diag + this->kinetic_energy_matrix());
        } else {
          row_result = arma::join_rows(row_result, potential_diag);
        }
      }

      result = arma::join_cols(result, row_result);
    }
    return result;
  }


  inline
  arma::vec positional_expectation() const {
    arma::vec result = arma::vec(this->dim());
#pragma omp parallel for
    for (arma::uword i = 0; i < result.n_elem; i++) {
      const cx_double dimension_result =
          arma::cdot(this->coefs,
                     this->positional_matrices.slice(i) * this->coefs);
      assert(std::abs(dimension_result.imag()) < 1e-6);
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
      assert(std::abs(dimension_result.imag()) < 1e-6);
      result(i) = std::real(dimension_result);
    }

    return result / this->norm() / this->norm();
  }

  inline
  std::vector<cwa::State> wigner_transform(
      const arma::uvec & momentum_space_grid,
      const arma::mat & momentum_space_ranges) const {

    if(momentum_space_grid.n_elem != momentum_space_ranges.n_rows) {
      throw Error("Different dimension between the grid and range provided");
    }

    const arma::uvec phase_space_grid = arma::join_cols(this->grid, momentum_space_grid);
    const arma::mat phase_space_range = arma::join_cols(this->ranges,
                                                        momentum_space_ranges);
    const arma::uvec phase_space_table = math::space::grids_to_table(
        phase_space_grid);
    const arma::uvec real_space_table = math::space::grids_to_table(this->grid);
    const arma::umat phase_space_iterations =
        math::space::auto_iteration_over_dims(phase_space_grid);
    const arma::umat Y_iterations =
        math::space::auto_iteration_over_dims(this->grid / 2 + 1);
    const arma::mat phase_space_points =
        math::space::points_generate(phase_space_grid, phase_space_range);
    const arma::vec scaling =
        (phase_space_range.col(1) - phase_space_range.col(0)) / (phase_space_grid - 1);

    std::vector<cwa::State> all_states;
    const arma::cx_mat split_up_coefs =
        arma::reshape(this->coefs, this->coefs.n_elem / this->layers, this->layers);

    for(arma::uword i_layer=0; i_layer <this->layers; i_layer++) {

      arma::vec weights(phase_space_points.n_cols, arma::fill::zeros);

      const arma::cx_vec layer_coefs = split_up_coefs.col(i_layer);
      #pragma omp parallel for
      for (arma::uword i = 0; i < weights.n_elem; i++) {

        const arma::uvec X = phase_space_iterations.col(i).rows(0, this->dim()-1);

        const arma::vec P = phase_space_points.col(i).rows(this->dim(), 2*this->dim()-1);

        for (arma::uword j = 0; j < Y_iterations.n_cols; j++) {
          const arma::uvec Y = Y_iterations.col(j);
          const arma::vec Y_num = Y % scaling.rows(0, this->dim() - 1);

          const arma::uvec X_less_than_Y = arma::find(X<Y);
          const arma::uvec X_plus_Y_greater_than_grid = arma::find(X + Y > this->grid - 1);

          if(X_less_than_Y.n_elem == 0 && X_plus_Y_greater_than_grid.n_elem == 0) {
            const arma::uvec X_minus_Y = X-Y;
            const arma::uvec X_plus_Y = X+Y;
            const arma::uword X_minus_Y_index =
                math::space::indices_to_index(X_minus_Y,real_space_table);
            const arma::uword X_plus_Y_index =
                math::space::indices_to_index(X_plus_Y,real_space_table);

            const arma::uvec non_zero_Y = arma::find(Y);
            if(non_zero_Y.n_elem == 0) {
              const double term = std::real(
                  std::exp(- 2.0 * cx_double{0.0,1.0} * arma::dot(P,Y_num)) *
                  std::conj(this->coefs(X_minus_Y_index)) * this->coefs(X_plus_Y_index));

              weights(i) += term / std::pow(2.0 * math::pi, this->dim());
            }
            else {
              const double term = 2.0 * std::real(
                  std::exp(- 2.0 * cx_double{0.0,1.0} * arma::dot(P,Y_num)) *
                  std::conj(this->coefs(X_minus_Y_index)) * this->coefs(X_plus_Y_index));

              weights(i) += term / std::pow(2.0 * math::pi, this->dim());
            }
          }
        }
      }

      all_states.push_back(cwa::State(phase_space_points, weights, this->masses));
    }

    return all_states;

  }

  template<typename Function>
  std::vector<arma::vec> expectation(const std::vector<Function> & observables) const {
    const std::vector<cwa::State> transformed = this->wigner_transform();

    std::vector<arma::vec> result(this->layers);

    for(auto i=0; i<this->layers; i++) {
      result[i] = transformed[i].expectation(observables);
    }

    return result;
  }

  template<typename Function>
  double expectation(const Function & observable) const {
    const std::vector<cwa::State> transformed = this->wigner_transform();

    std::vector<arma::vec> result(this->layers);

    for(auto i=0; i<this->layers; i++) {
      result[i] = transformed[i].expectation(observable);
    }

    return result;
  }

  inline
  std::vector<cwa::State> wigner_transform() const {
    return this->wigner_transform(this->grid, this->ranges);
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

  State operator()(State state) const {
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

#endif //M_DVR_H