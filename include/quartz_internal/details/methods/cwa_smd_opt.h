#ifndef METHODS_CWA_SMD_OPT_H
#define METHODS_CWA_SMD_OPT_H

#include <gsl/gsl_multimin.h>
#include "quartz_internal/util/gsl_converter.h"

namespace method {
namespace cwa_smd_opt {
namespace details {

inline
gsl_multimin_fdfminimizer_type * minimizer_map(const std::string type) {
  if(type == "steepest_descent") {
    return const_cast<gsl_multimin_fdfminimizer_type *>(gsl_multimin_fdfminimizer_steepest_descent);
  } else if(type == "conjugate_pr") {
    return const_cast<gsl_multimin_fdfminimizer_type *>(gsl_multimin_fdfminimizer_conjugate_pr);
  } else if(type == "conjugate_fr") {
    return const_cast<gsl_multimin_fdfminimizer_type *>(gsl_multimin_fdfminimizer_conjugate_fr);
  } else if(type == "bfgs") {
    return const_cast<gsl_multimin_fdfminimizer_type *>(gsl_multimin_fdfminimizer_vector_bfgs);
  } else if(type == "bfgs2") {
    return const_cast<gsl_multimin_fdfminimizer_type *>(gsl_multimin_fdfminimizer_vector_bfgs2);
  } else {
    throw Error("minimizer " + type + " is not implemented");
  }

}

struct cwa_smd_opt_param {
  arma::mat original_points;
  arma::vec expectations_ref;
  std::vector<math::Polynomial<double>> original_operators;
  arma::vec weights;
  arma::vec scaling;
  long long grade;
};


inline
double penalty_function(
    const arma::mat & points,
    const arma::vec & expectations_ref,
    const std::vector<math::Polynomial<double>> & original_operators,
    const arma::vec & weights,
    const arma::vec & scaling,
    const long long grade) {

  double result = 0;

  for (arma::uword i = 0; i < original_operators.size(); i++) {
    const long long the_grade = original_operators[i].grade();
    if (the_grade < grade && the_grade > 0) {
      const double result_from_cwa =
          cwa_smd::details::expectation(original_operators[i], points, weights,
                                        scaling);

      result += std::pow(result_from_cwa - expectations_ref(i), 2);
    }
  }

  return result;

}

inline
arma::mat penalty_function_derivative(
    const arma::mat & points,
    const arma::vec & expectations_ref,
    const std::vector<math::Polynomial<double>> & original_operators,
    const arma::vec & weights,
    const arma::vec & scaling,
    const long long grade
) {

  arma::mat result(arma::size(points), arma::fill::zeros);

  for (arma::uword i = 0; i < original_operators.size(); i++) {
    const long long the_grade = original_operators[i].grade();
    if (the_grade < grade && the_grade > 0) {
      const double result_from_cwa =
          cwa_smd::details::expectation(original_operators[i], points, weights,
                                        scaling);

      for (arma::uword j = 0; j < points.n_cols; j++) {
        const arma::vec point = arma::diagmat(1.0 / scaling) * points.col(j);
        for (arma::uword k = 0; k < points.n_rows / 2; k++) {
          const math::Polynomial<double> x_derivative =
              original_operators[i].derivative(
                  k) / scaling(k);
          const math::Polynomial<double> p_derivative =
              original_operators[i].derivative(
                  k + points.n_rows / 2) / scaling(k + points.n_rows / 2);

          result(k, j) -=
              2.0 * (result_from_cwa - expectations_ref(i)) * weights(j) *
              x_derivative.at(point) / arma::sum(weights);
          result(k + points.n_rows / 2, j) -=
              2.0 * (result_from_cwa - expectations_ref(i)) * weights(j) *
              p_derivative.at(point) / arma::sum(weights);
        }
      }
    }
  }

  return -result;

}

inline
double penalty_function_gsl_wrapper(const gsl_vector * flattened_points,
                                    void * param) {

  const arma::vec arma_flattened_points = gsl::convert_vec(flattened_points);
  const auto converted_param = *(cwa_smd_opt_param *) param;
  const arma::mat points = arma::reshape(arma_flattened_points, arma::size(
      converted_param.original_points));

  return penalty_function(points,
                          converted_param.expectations_ref,
                          converted_param.original_operators,
                          converted_param.weights,
                          converted_param.scaling,
                          converted_param.grade);
}

inline
void penalty_function_derivative_gsl_wrapper(
    const gsl_vector * flattened_points,
    void * param,
    gsl_vector * g) {
  const arma::vec arma_flattened_points = gsl::convert_vec(flattened_points);
  const auto converted_param = *(cwa_smd_opt_param *) param;
  const arma::mat points = arma::reshape(arma_flattened_points, arma::size(
      converted_param.original_points));

  const arma::vec result =
      arma::vectorise(
          penalty_function_derivative(points,
                                      converted_param.expectations_ref,
                                      converted_param.original_operators,
                                      converted_param.weights,
                                      converted_param.scaling,
                                      converted_param.grade));

  const auto result_pointer = gsl::convert_vec(result);

  gsl_vector_memcpy(g, result_pointer);

  gsl_vector_free(result_pointer);
}

inline
void penalty_function_fdf_gsl_wrapper(
    const gsl_vector * a_derivatives,
    void * param,
    double * f,
    gsl_vector * g) {
  penalty_function_derivative_gsl_wrapper(a_derivatives, param, g);

  *f = penalty_function_gsl_wrapper(a_derivatives, param);
}

inline
std::tuple<arma::mat, double, double, int> cwa_optimize(const cwa_smd_opt_param input,
                                                const double initial_step_size,
                                                const double tolerance,
                                                const double gradient_tolerance,
                                                const size_t total_steps,
                                                const std::string type) {

  /* allocate memory for minimization process */
  const auto minimizer_type = minimizer_map(type);

  const arma::uword n = input.original_points.n_elem;

  const auto penalty_function_value = penalty_function(input.original_points,
                                                       input.expectations_ref,
                                                       input.original_operators,
                                                       input.weights,
                                                       input.scaling,
                                                       input.grade);

  const double gradient_module = arma::norm(
      arma::vectorise(penalty_function_derivative(input.original_points,
                                                  input.expectations_ref,
                                                  input.original_operators,
                                                  input.weights,
                                                  input.scaling,
                                                  input.grade)));

  if (penalty_function_value < tolerance &&
      gradient_module < gradient_tolerance) {
    return {input.original_points, penalty_function_value, gradient_module, 0};
  }

  auto minimizer_environment = gsl_multimin_fdfminimizer_alloc(minimizer_type,
                                                               n);

  /* assigning function to minimizer object */
  gsl_multimin_function_fdf minimizer_object;
  minimizer_object.f = &penalty_function_gsl_wrapper;
  minimizer_object.df = &penalty_function_derivative_gsl_wrapper;
  minimizer_object.fdf = &penalty_function_fdf_gsl_wrapper;
  minimizer_object.n = n;
  minimizer_object.params = (void *) &input;

  /* starting point */
  const arma::vec flattened = arma::vectorise(input.original_points);
  gsl_vector * points = gsl::convert_vec(flattened);

  /* set environment */
  gsl_multimin_fdfminimizer_set(minimizer_environment,
                                &minimizer_object, points,
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

      const double f = minimizer_environment->f;
      const double df = arma::norm(gsl::convert_vec(minimizer_environment->gradient));
      gsl_multimin_fdfminimizer_free(minimizer_environment);
      gsl_vector_free(points);

      return {arma::reshape(result, arma::size(input.original_points)), f, df, iter};
    }
  } while (status == GSL_CONTINUE && iter < total_steps);

  throw Error("fail to converge towards the solution");
}

} // namespace details

struct State {
public:
  arma::mat points;
  arma::vec weights;
  arma::vec masses;
  arma::uword grade;
  arma::uvec expectation_table;
  arma::vec expectations;
  arma::uvec positional_indices;
  arma::uvec momentum_indices;
  arma::vec scaling;

  // Establish an easy way to construct your State
  template<typename PhaseSpaceDistribution>
  State(const PhaseSpaceDistribution & initial,
        const arma::uvec & grid,
        const arma::mat & range,
        const arma::vec & scaling,
        const arma::vec & masses,
        const arma::uword grade) :
      points(math::space::points_generate(grid, range)),
      weights(arma::real(at(initial, points))),
      masses(masses),
      grade(grade),
      expectation_table(math::space::grids_to_table(
          grade * arma::ones<arma::uvec>(points.n_rows))),
      scaling(scaling) {
    if (grid.n_rows != range.n_rows) {
      throw Error("Different dimension between the grid and the range");
    }
    if (grid.n_rows != 2 * masses.n_rows) {
      throw Error("Different dimension between the grid and the masses");
    }

    const arma::uword dimension = grid.n_elem;
    const arma::uword length = std::pow(grade, dimension);

    this->expectations = arma::vec(length);
    this->positional_indices = arma::uvec(dimension / 2);
    this->momentum_indices = arma::uvec(dimension / 2);

    const arma::vec ranges = range.col(1) - range.col(0);

    // exponents check in
#pragma omp parallel for
    for (arma::uword i = 0; i < dimension / 2; i++) {
      arma::uvec X = arma::zeros<arma::uvec>(dimension);
      arma::uvec P = arma::zeros<arma::uvec>(dimension);
      X(i) = 1;
      P(i + dimension / 2) = 1;
      this->positional_indices(i) =
          math::space::indices_to_index(X, this->expectation_table);
      this->momentum_indices(i) =
          math::space::indices_to_index(P, this->expectation_table);
    }

    // expectations check in
#pragma omp parallel for
    for (arma::uword i = 0; i < length; i++) {
      const lvec indices =
          arma::conv_to<lvec>::from(
              math::space::index_to_indices(i, this->expectation_table));

      this->expectations(i) =
          cwa_smd::details::expectation(math::polynomial::Term(1.0, indices),
                                        this->points, this->weights,
                                        this->scaling);
    }
  }

  template<typename PhaseSpaceDistribution>
  State(const PhaseSpaceDistribution & initial,
        const arma::uvec & grid,
        const arma::mat & range,
        const arma::uword grade) :
      points(math::space::points_generate(grid, range)),
      weights(arma::real(at(initial, points))),
      masses(arma::ones<arma::vec>(grid.n_rows / 2)),
      grade(grade),
      expectation_table(math::space::grids_to_table(
          grade * arma::ones<arma::uvec>(points.n_rows))) {
    if (grid.n_rows != range.n_rows) {
      throw Error("Different dimension between the grid and the range");
    }
    if (grid.n_rows != 2 * masses.n_rows) {
      throw Error("Different dimension between the grid and the masses");
    }

    const auto dimension = grid.n_elem;
    const auto length = std::pow(grade, dimension);

    this->expectations = arma::vec(length);
    this->positional_indices = arma::uvec(dimension / 2);
    this->momentum_indices = arma::uvec(dimension / 2);

    const arma::vec ranges = range.col(1) - range.col(0);
    this->scaling = ranges;
//    this->scaling = arma::ones(arma::size(ranges));

    // exponents check in
    for (arma::uword i = 0; i < dimension / 2; i++) {
      arma::uvec X = arma::zeros<arma::uvec>(dimension);
      arma::uvec P = arma::zeros<arma::uvec>(dimension);
      X(i) = 1;
      P(i + dimension / 2) = 1;
      this->positional_indices(i) =
          math::space::indices_to_index(X, this->expectation_table);
      this->momentum_indices(i) =
          math::space::indices_to_index(P, this->expectation_table);
    }

    // expectations check in
    for (arma::uword i = 0; i < length; i++) {
      const lvec indices =
          arma::conv_to<lvec>::from(
              math::space::index_to_indices(i, this->expectation_table));

      this->expectations(i) =
          cwa_smd::details::expectation(
              math::polynomial::Term(1.0, indices),
              this->points, this->weights, this->scaling);
    }
  }

  inline
  State(const arma::mat & points,
        const arma::vec & weights,
        const arma::vec & masses,
        const arma::uvec & expectation_table,
        const arma::vec & expectations,
        const arma::uvec & positional_indices,
        const arma::uvec & momentum_indices,
        const arma::vec & scaling,
        const arma::uword grade) :
      points(points),
      weights(weights),
      masses(masses),
      grade(grade),
      expectation_table(expectation_table),
      expectations(expectations),
      positional_indices(positional_indices),
      momentum_indices(momentum_indices),
      scaling(scaling) {}

  inline
  State(const State & state) :
      points(state.points),
      weights(state.weights),
      masses(state.masses),
      grade(state.grade),
      expectation_table(state.expectation_table),
      expectations(state.expectations),
      positional_indices(state.positional_indices),
      momentum_indices(state.momentum_indices),
      scaling(state.scaling) {}

  inline
  arma::uword dim() const {
    return points.n_rows / 2;
  }

  inline
  State normalise() const {
    State state = *this;
    state.weights = state.weights / arma::sum(state.weights);

    return state;
  }

  inline
  arma::vec positional_expectation() const {

    const arma::vec result = this->expectations(this->positional_indices);
    const arma::vec scale = this->scaling.rows(0, this->dim() - 1);

    return result * scale;
  }

  inline
  arma::vec momentum_expectation() const {
    const arma::vec result = this->expectations(this->momentum_indices);
    const arma::vec scale = this->scaling.rows(this->dim(),
                                               2 * this->dim() - 1);

    return result * scale;
  }

  State operator+(const State & B) const {
    if (!arma::approx_equal(this->weights, B.weights, "abs_diff", 1e-16) ||
        !arma::approx_equal(this->masses, B.masses, "abs_diff", 1e-16)) {
      throw Error("Different cwa states are being added");
    }

    State state = B;
    state.points += this->points;
    state.expectations += this->expectations;

    return state;
  }

  State operator*(const double B) const {

    State state = *this;
    state.expectations *= B;
    state.points *= B;

    return state;
  }

  template<typename T>
  auto expectation(const math::Polynomial<T> & polynomial) const {
    return cwa_smd::details::at_search(polynomial,
                                       this->points,
                                       this->weights,
                                       this->expectations,
                                       this->expectation_table,
                                       this->scaling,
                                       this->grade) *
           polynomial.at(this->scaling);
  }

  template<typename T>
  arma::vec expectation(const std::vector<math::Polynomial<T>>

                        & polynomials) const {
    arma::vec result(polynomials.size());

#pragma omp parallel for
    for (arma::uword i = 0; i < result.n_elem; i++) {
      result(i) = this->expectation(polynomials[i]);
    }

    return result;
  }

  State & operator=(const State &) = default;
};

struct Operator {

public:
  math::Polynomial<double> potential;
  math::Polynomial<double> H;
  std::vector<math::Polynomial<double>> original_operators;
  std::vector<math::Polynomial<double>> operators;

  Operator(const State & state,
           const math::Polynomial<double> & potential) :
      potential(potential),
      H(hamiltonian(potential, state.masses).scale(state.scaling)),
      operators() {

    std::vector<math::Polynomial<double>>
        op(std::pow(state.grade, state.dim() * 2));
    std::vector<math::Polynomial<double>>
        original_op(std::pow(state.grade, state.dim() * 2));

    op[0] = math::Polynomial<double>(state.dim() * 2, 0.0);
    original_op[0] = math::Polynomial<double>(state.dim() * 2, 1.0);


    for (arma::uword i = 1; i < op.size(); i++) {
      const auto observable =
          math::Polynomial(math::polynomial::Term<double>(1.0,
                                                          math::space::index_to_indices(
                                                              i,
                                                              state.expectation_table)));

      original_op[i] = observable;

      const arma::uword cut_off = std::min(observable.grade(), H.grade()) / 2;
      const auto moyal =
          moyal_bracket(math::Polynomial(observable), H, state.scaling,
                        cut_off);

      op[i] = moyal;
    }

    this->operators = op;
    this->original_operators = original_op;
  }


  inline
  PropagationType propagation_type() const {
    return Classic;
  }

  State operator()(const State & state) const {

    arma::mat p_submatrix = state.points.rows(state.dim(), 2 * state.dim() - 1);
    p_submatrix.each_col() /= state.masses;

    const arma::mat points_change_list =
        arma::join_cols(p_submatrix,
                        cwa::details::force(this->potential,
                                            state.points.rows(0, state.dim() -
                                                                 1)));

    arma::vec expectation_change_list =
        arma::vec(arma::size(state.expectations));

#pragma omp parallel for
    for (arma::uword i = 0; i < expectation_change_list.n_elem; i++) {
      expectation_change_list(i) =
          cwa_smd::details::at_search(this->operators[i],
                                      state.points,
                                      state.weights,
                                      state.expectations,
                                      state.expectation_table,
                                      state.scaling,
                                      state.grade);
    }

    return State(points_change_list,
                 state.weights,
                 state.masses,
                 state.expectation_table,
                 expectation_change_list,
                 state.positional_indices,
                 state.momentum_indices,
                 state.scaling,
                 state.grade);
  }

};

template<typename Potential>
OperatorWrapper<Operator, State, Potential>
cwa_opt(const double initial_step_size,
        const double tolerance,
        const double gradient_tolerance,
        const size_t total_steps,
        const std::string type = "bfgs2",
        const int print_level = 0) {
  return [initial_step_size,
      tolerance,
      gradient_tolerance,
      total_steps,
      type,
      print_level
  ](const Operator & cwa_smd_opt_operator,
    const Potential & potential) -> Propagator<State> {
    return [initial_step_size,
        tolerance,
        gradient_tolerance,
        total_steps,
        &cwa_smd_opt_operator,
        type,
        print_level
    ]
        (const State & state,
         const double dt) -> State {
      const arma::vec & ref_expectations = state.expectations;
      const arma::mat & points = state.points;
      const auto & original_operators = cwa_smd_opt_operator.original_operators;

      details::cwa_smd_opt_param input{points, ref_expectations,
                                       original_operators,
                                       state.weights, state.scaling,
                                       (long long) state.grade};

      const auto opt_result = details::cwa_optimize(input, initial_step_size,
                                tolerance, gradient_tolerance, total_steps, type);

      const arma::mat new_points = std::get<0>(opt_result);
      const double f = std::get<1>(opt_result);
      const double df = std::get<2>(opt_result);
      const int iter = std::get<3>(opt_result);

      if(print_level > 2) {
        fmt::print("f: {0:20.10f}, df: {1:20.10f}, iter: {2}", f, df, iter);
        fmt::print("\n");
      }

      State new_state = state;
      new_state.points = new_points;
      return new_state;
    };
  };
}

} // namespace cwa
}

#endif //METHODS_CWA_SMD_OPT_H