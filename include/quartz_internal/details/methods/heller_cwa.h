#ifndef METHODS_HELLER_H
#define METHODS_HELLER_H

#include <gsl/gsl_multimin.h>

namespace method {
// use your method name to create a subspace for your
// implementation of details
namespace heller_cwa {
namespace details {

struct HellerParam {

  // exp(a[0] p^2 + a[1] p + a[2] p q + a[3] q + a[4] q^2 + a[5])
  // alpha = 0.5 / a[0]
  // delta = a[1]
  // epsilon = a[2]

  arma::cx_vec a;
  arma::mat points;
  math::Polynomial<cx_double> V_eff_0;
  double mass;
};

inline
cx_double heller_gaussian(const arma::cx_vec & a,
                          const double q,
                          const double p) {
  return std::exp(a(0) * std::pow(p, 2)
                  + a(1) * p
                  + a(2) * p * q
                  + a(3) * q
                  + a(4) * std::pow(q, 2)
                  + a(5));
}

inline
arma::cx_vec heller_gaussian(const arma::cx_vec & a,
                             const arma::mat & points) {
  if (points.n_rows != 2) {
    throw Error("heller_gaussian only supports one-dimensional phase space");
  }

  arma::cx_vec result(points.n_cols);
#pragma omp parallel for
  for (arma::uword i = 0; i < result.n_elem; i++) {
    result(i) = heller_gaussian(a, points(0, i), points(1, i));
  }

  return result;
}

inline
math::Gaussian<cx_double> heller_gaussian(const arma::cx_vec & a) {

  const arma::cx_mat binomial{{{-a(4) * 2.0, -a(2)}, {-a(2), -a(0) * 2.0}}};
  const arma::cx_mat covariance = arma::inv(binomial);
  const arma::cx_vec center = covariance * arma::cx_vec{a(3),a(1)};

  return math::Gaussian<cx_double>(covariance, center,
      std::exp(a(5)) * std::exp(0.5 * arma::dot(center, arma::cx_vec{a(3),a(1)})));

}

template<typename Potential>
arma::mat force(const Potential & potential,
                const arma::mat & positions) {
  arma::mat result = arma::mat(arma::size(positions));

#pragma omp parallel for
  for (arma::uword i = 0; i < positions.n_cols; i++) {
    const arma::vec position = positions.col(i);
    for (arma::uword j = 0; j < positions.n_rows; j++) {
      result(j, i) = -potential.derivative(j).at(position);
    }
  }

  return result;
}

template<typename Function>
inline
auto effective_potential(const Function & potential,
                         const cx_double alpha,
                         const cx_double delta,
                         const cx_double epsilon,
                         const arma::uword l,
                         const arma::uword cut_off) {

  if (potential.dim() > 1) {
    throw Error("Currently effective potential is only valid "
                "for one dimensional potential");
  }

  const auto functor = [alpha](const Function & function) -> Function {
    return function.derivative(0).derivative(0) / 8.0 / alpha;
  };

  const auto a = exp(functor, derivative(potential, arma::uvec{l}), cut_off) *
                 std::pow(-1, l + 1);
  const auto b = exp(functor, derivative(potential, arma::uvec{l}), cut_off);

  const std::vector<math::Polynomial<cx_double>>
      a_inner =
      {math::Polynomial<cx_double>(
          arma::cx_vec{1.0 + epsilon / 2.0,
                       cx_double{0.0, 1.0} / 2.0 / alpha,
                       0.5 * delta},
          lmat{{1, 0, 0},
               {0, 1, 0}})};

  const std::vector<math::Polynomial<cx_double>>
      b_inner =
      {math::Polynomial<cx_double>(
          arma::cx_vec{1.0 - epsilon / 2.0,
                       -cx_double{0.0, 1.0} / 2.0 / alpha,
                       -0.5 * delta},
          lmat{{1, 0, 0},
               {0, 1, 0}})};

  const auto combined = (a(a_inner) + b(b_inner)) * 0.5;

  return combined;
}

inline
math::Polynomial<cx_double> E_function(const arma::cx_vec & a,
                                       const arma::cx_vec & a_derivatives,
                                       const math::Polynomial<cx_double> & V_eff_0,
                                       const double mass) {

  if (a.n_elem != 6 || a_derivatives.n_elem != 6) {
    throw Error(
        "The number of elements for a's or their derivatives is invalid for E function");
  }

  const math::Polynomial<cx_double> without_derivatives(
      arma::cx_vec{-a(2) / mass, -2.0 * a(4) / mass, -a(3) / mass},
      lmat{{{0, 1, 0}, {2, 1, 1}}}
  );

  const math::Polynomial<cx_double> derivatives(
      arma::cx_vec{-a_derivatives(0),
                   -a_derivatives(1),
                   -a_derivatives(2),
                   -a_derivatives(3),
                   -a_derivatives(4),
                   -a_derivatives(5)},
      lmat{{{0, 0, 1, 1, 2, 0}, {2, 1, 1, 0, 0, 0}}}
  );

  return without_derivatives + derivatives - V_eff_0 * cx_double{0.0, 2.0};

}

inline
double I_function(
    const arma::cx_vec & a,
    const arma::cx_vec & a_derivatives,
    const arma::mat & points,
    const math::Polynomial<cx_double> & V_eff_0,
    const double mass) {

  return arma::sum(arma::pow(arma::abs(
      at(E_function(a, a_derivatives, V_eff_0, mass), points)
      % heller_gaussian(a, points)), 2));

}

inline
arma::vec I_function_derivative(
    const arma::cx_vec & a,
    const arma::cx_vec & a_derivatives,
    const arma::mat & points,
    const math::Polynomial<cx_double> & V_eff_0,
    const double mass
) {

  arma::vec result(12);

  const auto E_func = E_function(a, a_derivatives, V_eff_0, mass);

  const auto polynomial_term_list = lmat{{{0, 0, 1, 1, 2, 0}, {2, 1, 1, 0, 0, 0}}};

#pragma omp parallel for
  for (arma::uword i = 0; i < 6; i++) {

    const lvec E_derivative_term = polynomial_term_list.col(i);
    const auto E_derivative =
        math::polynomial::Term<double>(1.0, E_derivative_term);

    result(i) = -arma::sum(2.0 *
                           arma::real(at(E_func.conj() * E_derivative, points))
                           %
                           arma::pow(arma::abs(heller_gaussian(a, points)), 2));

    result(i + 6) = arma::sum(2.0 *
                              arma::imag(
                                  at(E_func.conj() * E_derivative, points))
                              %
                              arma::pow(arma::abs(heller_gaussian(a, points)),
                                        2));
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
                    heller_param.points,
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
                                             heller_param.points,
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
                        details::force(potential,
                                       state.points.rows(0, state.dim() - 1)));

    return State(change_list, state.weights, state.masses);
  }

};

} // namespace heller
}

#endif //METHODS_HELLER_H
