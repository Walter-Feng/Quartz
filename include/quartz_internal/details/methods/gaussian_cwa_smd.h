#ifndef METHODS_GAUSSIAN_CWA_SMD_H
#define METHODS_GAUSSIAN_CWA_SMD_H

namespace method {
namespace g_cwa_smd {
namespace details {


template<typename Function>
auto expectation(const Function & function,
                 const arma::mat & points,
                 const arma::vec & weights) {

  auto result = at(function, points);

  return arma::dot(result, weights) / arma::sum(weights);
}

template<typename T>
auto at_search(const math::polynomial::Term <T> & term,
               const math::Gaussian <T> & gaussian,
               const arma::mat & points,
               const arma::vec & weights,
               const arma::vec & expectations,
               const arma::uvec & table,
               const arma::uword grade) {

  if ((arma::uword) arma::max(term.exponents) >= grade) {
    return expectation(
        math::GaussianWithPoly(math::Polynomial(term), gaussian),
        points, weights);
  } else {
    const arma::uvec indices = arma::conv_to<arma::uvec>::from(term.exponents);

    return term.coef *
           expectations(math::space::indices_to_index(indices, table));
  }
}

template<typename T>
auto at_search(const math::Polynomial <T> & polynomial,
               const math::Gaussian <T> & gaussian,
               const arma::mat & points,
               const arma::vec & weights,
               const arma::vec & expectations,
               const arma::uvec & table,
               const arma::uword grade) {

  auto result = at_search(polynomial.term(0), gaussian, points, weights,
                          expectations,
                          table, grade);

  for (arma::uword i = 1; i < polynomial.coefs.n_elem; i++) {
    result += at_search(polynomial.term(i), gaussian, points, weights,
                        expectations,
                        table, grade);
  }

  return result;
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
  math::Gaussian<double> gaussian;
  math::Polynomial<double> taylor_expansion;

  // Establish an easy way to construct your State
  template<typename PhaseSpaceDistribution>
  State(const PhaseSpaceDistribution & initial,
        const arma::uvec & grid,
        const arma::mat & range,
        const arma::vec & masses,
        const arma::uword grade) :
      points(math::space::points_generate(grid, range)),
      weights(arma::real(at(initial, points))),
      masses(masses),
      grade(grade),
      expectation_table(math::space::grids_to_table(
          grade * arma::ones<arma::uvec>(points.n_rows))),
      gaussian(),
      taylor_expansion() {
    if (grid.n_rows != range.n_rows) {
      throw Error("Different dimension between the grid and the range");
    }
    if (grid.n_rows != 2 * masses.n_rows) {
      throw Error("Different dimension between the grid and the masses");
    }

    const arma::uword dimension = grid.n_elem;
    const arma::uword length = std::pow(grade, dimension);

    this->expectations = arma::vec(length);

    const arma::vec standard_deviation =
        (arma::abs(range.col(1)) + arma::abs(range.col(0))) / 2.;

    const arma::mat covariance = arma::diagmat(
        arma::pow(standard_deviation, 2));

    this->gaussian = math::Gaussian<double>(covariance, arma::zeros<arma::vec>(
        covariance.n_rows));

    this->taylor_expansion =
        math::taylor_expand<math::GaussianWithPoly<double>, double>(
            math::GaussianWithPoly<double>(-covariance),
            arma::zeros<arma::vec>(grid.n_elem),
            this->grade
        );


    // expectations check in
#pragma omp parallel for
    for (arma::uword i = 0; i < length; i++) {
      const lvec indices =
          arma::conv_to<lvec>::from(
              math::space::index_to_indices(i, this->expectation_table));

      this->expectations(i) =
          details::expectation(
              math::GaussianWithPoly(
                  math::Polynomial(math::polynomial::Term(1.0, indices)),
                  this->gaussian),
              this->points, this->weights);
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
          grade * arma::ones<arma::uvec>(points.n_rows))),
      gaussian(initial.cov(), initial.mean()) {
    if (grid.n_rows != range.n_rows) {
      throw Error("Different dimension between the grid and the range");
    }
    if (grid.n_rows != 2 * masses.n_rows) {
      throw Error("Different dimension between the grid and the masses");
    }

    const arma::uword dimension = grid.n_elem;
    const arma::uword length = std::pow(grade, dimension);

    this->expectations = arma::vec(length);

    const arma::vec standard_deviation =
        (arma::abs(range.col(1)) + arma::abs(range.col(0))) / 2.;

    const arma::mat covariance = arma::diagmat(
        arma::pow(standard_deviation, 2));

    this->gaussian = math::Gaussian<double>(covariance, arma::zeros<arma::vec>(
        covariance.n_rows));

    this->taylor_expansion =
        math::taylor_expand<math::GaussianWithPoly<double>, double>(
            math::GaussianWithPoly<double>(-covariance),
            arma::zeros<arma::vec>(grid.n_elem),
            this->grade
        );


    // expectations check in
#pragma omp parallel for
    for (arma::uword i = 0; i < length; i++) {
      const lvec indices =
          arma::conv_to<lvec>::from(
              math::space::index_to_indices(i, this->expectation_table));

      this->expectations(i) =
          details::expectation(
              math::GaussianWithPoly(
                  math::Polynomial(math::polynomial::Term(1.0, indices)),
                  this->gaussian),
              this->points, this->weights);
    }
  }

  inline
  State(const arma::mat & points,
        const arma::vec & weights,
        const arma::vec & masses,
        const arma::uvec & expectation_table,
        const arma::vec & expectations,
        const math::Gaussian<double> & gaussian,
        const math::Polynomial<double> & taylor_expansion,
        const arma::uword grade) :
      points(points),
      weights(weights),
      masses(masses),
      grade(grade),
      expectation_table(expectation_table),
      expectations(expectations),
      gaussian(gaussian),
      taylor_expansion(taylor_expansion) {}

  inline
  State(const State & state) :
      points(state.points),
      weights(state.weights),
      masses(state.masses),
      grade(state.grade),
      expectation_table(state.expectation_table),
      expectations(state.expectations),
      gaussian(state.gaussian),
      taylor_expansion(state.taylor_expansion) {}

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

    arma::vec result(this->dim());

#pragma omp parallel for
    for (arma::uword i = 0; i < this->dim(); i++) {
      auto taylor_expansion_multiplied_by_x =
          this->taylor_expansion;
      taylor_expansion_multiplied_by_x.exponents.row(i) += 1;

      result(i) = details::at_search(taylor_expansion_multiplied_by_x,
                                     this->gaussian, this->points,
                                     this->weights, this->expectations,
                                     this->expectation_table, this->grade);
    }


    return result;
  }

  inline
  arma::vec momentum_expectation() const {
    arma::vec result(this->dim());

#pragma omp parallel for
    for (arma::uword i = 0; i < this->dim(); i++) {
      auto taylor_expansion_multiplied_by_p =
          this->taylor_expansion;
      taylor_expansion_multiplied_by_p.exponents.row(this->dim() + i) += 1;

      result(i) = details::at_search(taylor_expansion_multiplied_by_p,
                                     this->gaussian, this->points,
                                     this->weights, this->expectations,
                                     this->expectation_table, this->grade);
    }


    return result;
  }

  State operator+(const State & B) const {
    if (!arma::approx_equal(this->weights, B.weights, "abs_diff", 1e-16) ||
        !arma::approx_equal(this->masses, B.masses, "abs_diff", 1e-16)) {
      throw Error("Different md states are being added");
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
  auto expectation(const math::Polynomial <T> & polynomial) const {
    return details::at_search(polynomial,
                              this->gaussian,
                              this->points,
                              this->weights,
                              this->expectations,
                              this->expectation_table,
                              this->grade);
  }

  template<typename T>
  arma::vec expectation(const std::vector<math::Polynomial<T>> & observables) const {
    arma::vec result(observables.size());

#pragma omp parallel for
    for(arma::uword i=0; i<result.n_elem; i++) {
      result[i] = this->expectation(observables[i]);
    }

    return result;
  }
};

struct Operator {

public:
  math::Polynomial<double> potential;
  math::Polynomial<double> H;
  std::vector<math::Polynomial < double>> operators;

  Operator(const State & state,
           const math::Polynomial<double> & potential) :
      potential(potential),
      H(hamiltonian(potential, state.masses)),
      operators() {

    std::vector<math::Polynomial<double>>
        op(std::pow(state.grade, state.dim() * 2));

    op[0] = math::Polynomial<double>(state.dim() * 2);

    for (arma::uword i = 1; i < op.size(); i++) {
      const auto observable =
          math::Polynomial(math::polynomial::Term<double>(1.0,
                                                          math::space::index_to_indices(
                                                              i,
                                                              state.expectation_table)));

      const arma::uword cut_off = H.grade() / 2;
      const auto moyal =
          moyal_bracket(
              math::GaussianWithPoly(math::Polynomial(observable),state.gaussian),
              H, cut_off);

      op[i] = moyal.polynomial;
    }

    this->operators = op;
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
                                       state.points.rows(0, state.dim() - 1)));

    arma::vec expectation_change_list =
        arma::vec(arma::size(state.expectations));

#pragma omp parallel for
    for (arma::uword i = 0; i < expectation_change_list.n_elem; i++) {
      expectation_change_list(i) =
          details::at_search(this->operators[i],
                             state.gaussian,
                             state.points,
                             state.weights,
                             state.expectations,
                             state.expectation_table,
                             state.grade);
    }

    return State(points_change_list,
                 state.weights,
                 state.masses,
                 state.expectation_table,
                 expectation_change_list,
                 state.gaussian,
                 state.taylor_expansion,
                 state.grade);
  }

};

} // namespace g_cwa_smd
}

#endif //METHODS_GAUSSIAN_CWA_SMD_H
