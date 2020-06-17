#ifndef METHODS_HELLER_GAUSSIAN_H
#define METHODS_HELLER_GAUSSIAN_H

#include "heller_cwa.h"

namespace method {
// use your method name to create a subspace for your
// implementation of details
namespace heller_gaussian {
namespace details {

struct HellerParam {

  // exp(a[0] p^2 + a[1] p + a[2] p q + a[3] q + a[4] q^2 + a[5])
  // alpha = 0.5 / a[0]
  // delta = Im[a[1]]
  // epsilon = Im[a[2]]
  // p_bar = Re[a[1] * a[0]]

  arma::cx_vec a;
  math::Polynomial <cx_double> V_eff_0;
  double mass;
};

inline
double I_function(
    const arma::cx_vec & a,
    const arma::cx_vec & a_derivatives,
    const math::Polynomial <cx_double> & V_eff_0,
    const double mass) {

  const auto E_func_abs = heller_cwa::details::E_function(a, a_derivatives,
                                                          V_eff_0, mass).abs();

  const auto E_func_squared = E_func_abs * E_func_abs;

  const auto gaussian_abs = heller_cwa::details::heller_gaussian(a).abs();
  const auto gaussian_squared = gaussian_abs * gaussian_abs;

  return gaussian_squared.integral(E_func_squared);
}

inline
arma::vec I_function_derivative(
    const arma::cx_vec & a,
    const arma::cx_vec & a_derivatives,
    const math::Polynomial <cx_double> & V_eff_0,
    const double mass
) {

  arma::vec result(12);

  const auto E_func = heller_cwa::details::E_function(a, a_derivatives, V_eff_0,
                                                      mass);

  const auto polynomial_term_list = lmat{{{0, 0, 1, 1, 2, 0}, {2, 1, 1, 0, 0, 0}}};

  const auto gaussian_abs = heller_cwa::details::heller_gaussian(a).abs();
  const auto gaussian_squared = gaussian_abs * gaussian_abs;

#pragma omp parallel for
  for (arma::uword i = 0; i < 6; i++) {

    const lvec E_derivative_term = polynomial_term_list.col(i);
    const auto E_derivative =
        math::polynomial::Term<double>(1.0, E_derivative_term);

    result(i) = -2.0 * std::real(
        gaussian_squared.integral(E_func.conj() * E_derivative));

    result(i + 6) = 2.0 * std::imag(
        gaussian_squared.integral(E_func.conj() * E_derivative));
  }

  return result;
}

inline
double I_function_gsl_wrapper(
    const gsl_vector * a_derivatives,
    void * param
) {
  const HellerParam heller_param = *(HellerParam *) param;
  const arma::cx_vec a = heller_param.a;
  const arma::vec a_derivatives_all = gsl::convert_vec(a_derivatives);
  const arma::cx_vec a_derivatives_arma =
      arma::cx_vec{a_derivatives_all.rows(0, 5),
                   a_derivatives_all.rows(6, 11)};

  return I_function(a,
                    a_derivatives_arma,
                    heller_param.V_eff_0,
                    heller_param.mass);

}

inline
void I_function_derivative_gsl_wrapper(
    const gsl_vector * a_derivatives,
    void * param,
    gsl_vector * g
) {
  const HellerParam heller_param = *(HellerParam *) param;
  const arma::cx_vec a = heller_param.a;
  const arma::vec a_derivatives_all = gsl::convert_vec(a_derivatives);

  const arma::cx_vec a_derivatives_arma =
      arma::cx_vec{a_derivatives_all.rows(0, 5),
                   a_derivatives_all.rows(6, 11)};

  const auto result_pointer =
      gsl::convert_vec(I_function_derivative(a,
                                             a_derivatives_arma,
                                             heller_param.V_eff_0,
                                             heller_param.mass));

  gsl_vector_memcpy(g, result_pointer);

  gsl_vector_free(result_pointer);
}

inline
void I_function_fdf_gsl_wrapper(
    const gsl_vector * a_derivatives,
    void * param,
    double * f,
    gsl_vector * g
) {
  I_function_derivative_gsl_wrapper(a_derivatives, param, g);

  *f = I_function_gsl_wrapper(a_derivatives, param);
}


inline
arma::cx_vec a_derivative(HellerParam input,
                          const double initial_step_size,
                          const double tolerance,
                          const double gradient_tolerance,
                          const size_t total_steps) {

  /* allocate memory for minimization process */
  const auto minimizer_type = gsl_multimin_fdfminimizer_vector_bfgs2;
  auto minimizer_environment = gsl_multimin_fdfminimizer_alloc(minimizer_type,
                                                               12);

  /* assigning function to minimizer object */
  gsl_multimin_function_fdf minimizer_object;
  minimizer_object.f = &I_function_gsl_wrapper;
  minimizer_object.df = &I_function_derivative_gsl_wrapper;
  minimizer_object.fdf = &I_function_fdf_gsl_wrapper;
  minimizer_object.n = 12;
  minimizer_object.params = (void *) &input;

  /* starting point */
  const auto a_derivatives = gsl_vector_calloc(12);

  /* set environment */
  gsl_multimin_fdfminimizer_set(minimizer_environment,
                                &minimizer_object, a_derivatives,
                                initial_step_size, tolerance);

  size_t iter = 0;
  int status = GSL_CONTINUE;
  do {
    iter++;
    status = gsl_multimin_fdfminimizer_iterate(minimizer_environment);

    if (status) {
      throw Error(gsl_strerror(status));
    }

    status = gsl_multimin_test_gradient(minimizer_environment->gradient,
                                        gradient_tolerance);

    if (status == GSL_SUCCESS) {
      const arma::vec result = gsl::convert_vec(minimizer_environment->x);

      gsl_multimin_fdfminimizer_free(minimizer_environment);
      gsl_vector_free(a_derivatives);

      return arma::cx_vec{result.rows(arma::span(0, 5)),
                          result.rows(arma::span(6, 11))};
    }
  } while (status == GSL_CONTINUE && iter < total_steps);

  return arma::ones<arma::cx_vec>(6) * 2.8375;
}

} // namespace details

struct State {
public:
  arma::mat points;
  arma::vec weights;
  arma::vec masses;

  // Establish an easy way to construct your State
  template<typename PhaseSpaceDistribution>
  State(const PhaseSpaceDistribution & initial,
        const arma::uvec & grid,
        const arma::mat & range,
        const arma::vec & masses) :
      points(math::space::points_generate(grid, range)),
      weights(arma::real(at(initial, points))),
      masses(masses) {
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
        const arma::vec & masses) :
      points(points),
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
  State normalise() const {
    return State(this->points, this->weights / arma::sum(this->weights),
                 this->masses);
  }

  template<typename Function>
  arma::vec expectation(const std::vector<Function> & function) const {
    arma::vec result(function.size());

#pragma omp parallel for
    for (arma::uword i = 0; i < result.n_elem; i++) {

      if (function[i].dim() != this->dim() * 2) {
        throw Error(
            "The dimension of the function is invalid for the calculation of expectation");
      }
      result(i) = arma::dot(at(function[i], this->points), weights) /
                  arma::sum(weights);
    }

    return result;
  }

  template<typename Function>
  double expectation(const Function & function) const {
    const arma::vec result = at(function, this->points);

    return arma::dot(result, weights) / arma::sum(weights);
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

  State operator+(const State & B) const {
    if (!arma::approx_equal(this->weights, B.weights, "abs_diff", 1e-16) ||
        !arma::approx_equal(this->masses, B.masses, "abs_diff", 1e-16)) {
      throw Error("Different cwa states are being added");
    }

    return State(this->points + B.points, this->weights, this->masses);
  }

  State operator*(const double B) const {
    return State(this->points * B, this->weights, this->masses);
  }
};

template<typename Potential>
struct Operator {

private:
  PropagationType type = Classic;

public:
  Potential potential;

  Operator(const State & state,
           const Potential & potential) :
      potential(potential) {}


  inline
  PropagationType propagation_type() const {
    return Classic;
  }

  State operator()(const State & state) const {

    arma::mat p_submatrix = state.points.rows(state.dim(), 2 * state.dim() - 1);
    p_submatrix.each_col() /= state.masses;

    const arma::mat change_list =
        arma::join_cols(p_submatrix,
                        cwa::details::force(potential,
                                            state.points.rows(0, state.dim() -
                                                                 1)));

    return State(change_list, state.weights, state.masses);
  }

};

} // namespace heller
}

#endif //METHODS_HELLER_GAUSSIAN_H
