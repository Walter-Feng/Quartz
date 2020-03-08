#ifndef METHODS_CWA_SMD_H
#define METHODS_CWA_SMD_H

namespace method {
namespace cwa_smd {
namespace details {

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
auto expectation(const Function & function,
                 const arma::mat & points,
                 const arma::vec & weights) {

  auto result = at(function, points);

  return arma::dot(result, weights) / arma::sum(weights);
}

template<typename T>
auto at_search(const math::polynomial::Term <T> & term,
               const arma::mat & points,
               const arma::vec & weights,
               const arma::vec & expectations,
               const arma::uvec & table,
               const arma::uword grade) {

  if ((arma::uword) arma::max(term.indices) >= grade) {
    return expectation(term, points, weights);
  } else {
    const arma::uvec indices = arma::conv_to<arma::uvec>::from(term.indices);

    return term.coef *
           expectations(math::space::indices_to_index(indices, table));
  }
}

template<typename T>
auto at_search(const math::Polynomial <T> & polynomial,
               const arma::mat & points,
               const arma::vec & weights,
               const arma::vec & expectations,
               const arma::uvec & table,
               const arma::uword grade) {

  auto result = at_search(polynomial.term(0), points, weights, expectations,
                          table, grade);

  for (arma::uword i = 1; i < polynomial.coefs.n_elem; i++) {
    result += at_search(polynomial.term(i), points, weights, expectations,
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
  arma::uvec positional_indices;
  arma::uvec momentum_indices;

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
          grade * arma::ones<arma::uvec>(points.n_rows))) {
    if (grid.n_rows != range.n_rows) {
      throw Error("Different dimension between the grid and the range");
    }
    if (grid.n_rows != 2 * masses.n_rows) {
      throw Error("Different dimension between the grid and the masses");
    }

    const arma::uword dimension = grid.n_elem;
    const arma::uword length = std::pow(grade,dimension);

    this->expectations = arma::vec(length);
    this->positional_indices = arma::uvec(dimension / 2);
    this->momentum_indices = arma::uvec(dimension / 2);

    // indices check in
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

      this->expectations(i) = details::expectation(math::polynomial::Term(1.0,
                                                                          indices),
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
          grade * arma::ones<arma::uvec>(points.n_rows))) {
    if (grid.n_rows != range.n_rows) {
      throw Error("Different dimension between the grid and the range");
    }
    if (grid.n_rows != 2 * masses.n_rows) {
      throw Error("Different dimension between the grid and the masses");
    }

    const auto dimension = grid.n_elem;
    const auto length = std::pow(grade,dimension);

    this->expectations = arma::vec(length);
    this->positional_indices = arma::uvec(dimension / 2);
    this->momentum_indices = arma::uvec(dimension / 2);

    // indices check in
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

      this->expectations(i) = details::expectation(math::polynomial::Term(1.0,
                                                                          indices),
                                                   this->points, this->weights);
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
        const arma::uword grade) :
      points(points),
      weights(weights),
      masses(masses),
      grade(grade),
      expectation_table(expectation_table),
      expectations(expectations),
      positional_indices(positional_indices),
      momentum_indices(momentum_indices) {}

  inline
  State(const State & state) :
      points(state.points),
      weights(state.weights),
      masses(state.masses),
      grade(state.grade),
      expectation_table(state.expectation_table),
      expectations(state.expectations),
      positional_indices(state.positional_indices),
      momentum_indices(state.momentum_indices) {}

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

    return result;
  }

  inline
  arma::vec momentum_expectation() const {
    const arma::vec result = this->expectations(this->momentum_indices);

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
  auto expectation(const math::Polynomial <T> & polynomial) {
    return details::at_search(polynomial,
                              this->points,
                              this->weights,
                              this->expectations,
                              this->expectation_table,
                              this->grade);
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

#pragma omp parallel for
    for (arma::uword i = 1; i < op.size(); i++) {
      const auto observable =
          math::Polynomial(math::polynomial::Term<double>(1.0,
                                                          math::space::index_to_indices(
                                                              i,
                                                              state.expectation_table)));

      const auto cut_off = std::min(observable.grade(), H.grade());
      const auto moyal = moyal_bracket(math::Polynomial(observable), H,
                                       cut_off);

      op[i] = moyal;
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
                        details::force(this->potential,
                                       state.points.rows(0, state.dim() - 1)));

    arma::vec expectation_change_list =
        arma::vec(arma::size(state.expectations));

#pragma omp parallel for
    for (arma::uword i = 0; i < expectation_change_list.n_elem; i++) {
      expectation_change_list(i) =
          details::at_search(this->operators[i],
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
                 state.positional_indices,
                 state.momentum_indices,
                 state.grade);
  }

};

} // namespace md
}

#endif //METHODS_CWA_SMD_H
