#ifndef METHODS_DVR_SMD_H
#define METHODS_DVR_SMD_H

namespace method {
namespace dvr_smd {
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
                 const arma::vec & weights,
                 const arma::vec & scaling) {

  const arma::mat scaled_points = arma::diagmat(1 / scaling) * points;

  auto result = at(function, scaled_points);

  return arma::dot(result, weights) / arma::sum(weights);
}

template<typename Function>
auto expectation(const Function & function,
                 const cwa::State & state,
                 const arma::vec & scaling) {

  return expectation(function, state.points, state.weights, scaling);
}

template<typename T>
auto at_search(const math::polynomial::Term <T> & term,
               const cwa::State & state,
               const arma::vec & expectations,
               const arma::uvec & table,
               const arma::vec & scaling,
               const arma::uword grade) {

  if ((arma::uword) arma::max(term.exponents) >= grade) {
    return expectation(term, state.points, state.weights, scaling);
  } else {
    const arma::uvec indices = arma::conv_to<arma::uvec>::from(term.exponents);

    return term.coef *
           expectations(math::space::indices_to_index(indices, table));
  }
}

template<typename T>
auto at_search(const math::Polynomial <T> & polynomial,
               const cwa::State & state,
               const arma::vec & expectations,
               const arma::uvec & table,
               const arma::vec & scaling,
               const arma::uword grade) {

  auto result = at_search(polynomial.term(0), state, expectations,
                          table, scaling, grade);

  for (arma::uword i = 1; i < polynomial.coefs.n_elem; i++) {
    result += at_search(polynomial.term(i), state, expectations,
                        table, scaling, grade);
  }

  return result;
}


} // namespace details

struct State {
public:
  dvr::State dvr_state;
  arma::vec masses;
  arma::uword grade;
  arma::uvec expectation_table;
  arma::vec expectations;
  arma::uvec positional_indices;
  arma::uvec momentum_indices;
  arma::vec scaling;

  // Establish an easy way to construct your State
  template<typename WaveFunction>
  State(const WaveFunction & initial,
        const arma::uvec & grid,
        const arma::mat & range,
        const arma::vec & masses,
        const arma::uword grade) :
      dvr_state(initial,
                grid.rows(0, grid.n_elem / 2 - 1),
                range.rows(0, grid.n_elem / 2 - 1),
                masses),
      masses(masses),
      grade(grade),
      expectation_table(math::space::grids_to_table(
          grade * arma::ones<arma::uvec>(grid.n_elem))) {
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
    this->scaling = ranges;


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

      const cwa::State initial_cwa(initial.wigner_transform(), grid, range, masses);
      this->expectations(i) =
          details::expectation(math::polynomial::Term(1.0, indices),
                               initial_cwa, this->scaling);
    }
  }

  template<typename WaveFunction>
  State(const WaveFunction & initial,
        const arma::uvec & grid,
        const arma::mat & range,
        const arma::uword grade) :
      dvr_state(initial,
                grid.rows(0, grid.n_elem / 2 - 1),
                range.rows(0, grid.n_elem / 2 - 1),
                masses),
      masses(arma::ones<arma::vec>(grid.n_rows / 2)),
      grade(grade),
      expectation_table(math::space::grids_to_table(
          grade * arma::ones<arma::uvec>(grid.n_rows))) {
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

      const cwa::State initial_cwa(initial.wigner_transform(), grid, range, masses);

      this->expectations(i) =
          details::expectation(math::polynomial::Term(1.0, indices),
                               initial_cwa, this->scaling);
    }
  }

  inline
  State(const dvr::State & dvr_state,
        const arma::vec & masses,
        const arma::uvec & expectation_table,
        const arma::vec & expectations,
        const arma::uvec & positional_indices,
        const arma::uvec & momentum_indices,
        const arma::vec & scaling,
        const arma::uword grade) :
      dvr_state(dvr_state),
      masses(masses),
      grade(grade),
      expectation_table(expectation_table),
      expectations(expectations),
      positional_indices(positional_indices),
      momentum_indices(momentum_indices),
      scaling(scaling) {}

  inline
  State(const State & state) :
      dvr_state(state.dvr_state),
      masses(state.masses),
      grade(state.grade),
      expectation_table(state.expectation_table),
      expectations(state.expectations),
      positional_indices(state.positional_indices),
      momentum_indices(state.momentum_indices),
      scaling(state.scaling) {}

  inline
  arma::uword dim() const {
    return this->dvr_state.dim();
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
    State state = B;
    state.expectations += this->expectations;

    return state;
  }

  State operator*(const double B) const {

    State state = *this;
    state.expectations *= B;

    return state;
  }

  arma::vec expectation(const std::vector<math::Polynomial<double>> & polynomials) const {
    const auto transformed = this->dvr_state.wigner_transform();

    arma::vec result(polynomials.size());

#pragma omp parallel for
    for (arma::uword i = 0; i < result.n_elem; i++) {
      result(i) = details::at_search(polynomials[i],
                                     transformed,
                                     this->expectations,
                                     this->expectation_table,
                                     this->scaling,
                                     this->grade);
    }

    return result;
  }

  template<typename T>
  auto expectation(const math::Polynomial <T> & polynomial) const {
    return details::at_search(polynomial,
                              this->dvr_state.wigner_transform(),
                              this->expectations,
                              this->expectation_table,
                              this->scaling,
                              this->grade);
  }
};

struct Operator {

public:
  math::Polynomial<double> potential;
  math::Polynomial<double> H;
  std::vector<math::Polynomial < double>> operators;
  dvr::Operator dvr_operator;

  Operator(const State & state,
           const math::Polynomial<double> & potential) :
      potential(potential),
      H(hamiltonian(potential, state.masses).scale(state.scaling)),
      operators(),
      dvr_operator(state.dvr_state, potential) {

    std::vector<math::Polynomial<double>>
        op(std::pow(state.grade, state.dim() * 2));

    op[0] = math::Polynomial<double>(state.dim() * 2);

    for (arma::uword i = 1; i < op.size(); i++) {
      const auto observable =
          math::Polynomial(math::polynomial::Term<double>(1.0,
                                                          math::space::index_to_indices(
                                                              i,
                                                              state.expectation_table)));

      const arma::uword cut_off = std::min(observable.grade(), H.grade()) / 2;
      const auto moyal =
          moyal_bracket(math::Polynomial(observable), H, state.scaling,
                        cut_off);

      op[i] = moyal;
    }

    this->operators = op;
  }


  inline
  PropagationType propagation_type() const {
    return Mixed;
  }

  State operator()(const State & state) const {

    arma::vec expectation_change_list =
        arma::vec(arma::size(state.expectations));

#pragma omp parallel for
    for (arma::uword i = 0; i < expectation_change_list.n_elem; i++) {
      expectation_change_list(i) =
          details::at_search(this->operators[i],
                             state.dvr_state.wigner_transform(),
                             state.expectations,
                             state.expectation_table,
                             state.scaling,
                             state.grade);
    }

    return State(state.dvr_state,
                 state.masses,
                 state.expectation_table,
                 expectation_change_list,
                 state.positional_indices,
                 state.momentum_indices,
                 state.scaling,
                 state.grade);
  }

};

template<typename Operator, typename State, typename Potential>
OperatorWrapper <Operator, State, Potential>
    mixed_runge_kutta_4 = [](const Operator & liouville_operator,
                             const Potential & potential) -> Propagator <State> {

  static_assert(has_propagation_type<Operator, PropagationType(void)>::value,
                "Propagation type not specified");

  if (liouville_operator.propagation_type() != Mixed) {
    Error(
        "This wrapper is only valid for mixed type");
  }

  if constexpr(has_time_evolve<Potential, void(const double &)>::value) {
    return [&liouville_operator, &potential](const State & state,
                                             const double dt) -> State {

      Potential potential_at_half_dt = potential;
      potential_at_half_dt.time_evolve(0.5 * dt);

      Potential potential_at_dt = potential;
      potential_at_dt.time_evolve(dt);

      const Propagator<dvr::State>
          dvr_propagator =
          math::schrotinger_wrapper<dvr::Operator, dvr::State, Potential>(
              liouville_operator.dvr_operator, potential);
      const auto dvr_propagator_at_half_dt =
          math::schrotinger_wrapper<dvr::Operator, dvr::State, Potential>(
              liouville_operator.dvr_operator, potential_at_half_dt);
      const auto dvr_propagator_at_dt =
          math::schrotinger_wrapper<dvr::Operator, dvr::State, Potential>(
              liouville_operator.dvr_operator, potential_at_dt);

      const Operator operator_at_half_dt = Operator(state,
                                                    potential_at_half_dt);
      const Operator operator_at_dt = Operator(state, potential_at_dt);

      const State k1 = liouville_operator(state) * dt;
      State k1_with_dvr_at_half_dt = k1;
      k1_with_dvr_at_half_dt.dvr_state = dvr_propagator(k1.dvr_state, dt / 2.0);
      const State k2 =
          operator_at_half_dt(k1_with_dvr_at_half_dt * 0.5 + state) * dt;
      const State k3 =
          operator_at_half_dt(k1_with_dvr_at_half_dt * 0.5 + state) * dt;
      State k3_with_dvr_at_half_dt = k3;
      k3_with_dvr_at_half_dt.dvr_state = dvr_propagator_at_half_dt(k3.dvr_state,
                                                                   dt / 2.0);
      const State k4 = operator_at_dt(state + k3_with_dvr_at_half_dt) * dt;

      return state + k1 * (1.0 / 6.0) + k2 * (1.0 / 3.0) + k3 * (1.0 / 3.0) +
             k4 * (1.0 / 6.0);
    };
  } else {

    return [&liouville_operator, &potential](const State & state,
                                             const double dt) -> State {
      const auto dvr_propagator =
          math::schrotinger_wrapper<dvr::Operator, dvr::State, Potential>(
              liouville_operator.dvr_operator, potential);

      const State k1 = liouville_operator(state) * dt;
      State k1_with_dvr_at_half_dt = k1;
      k1_with_dvr_at_half_dt.dvr_state = dvr_propagator(k1.dvr_state, dt / 2.0);
      const State k2 = liouville_operator(k1 * 0.5 + state) * dt;
      const State k3 = liouville_operator(k2 * 0.5 + state) * dt;
      State k3_with_dvr_at_half_dt = k3;
      k3_with_dvr_at_half_dt.dvr_state = dvr_propagator(k3.dvr_state, dt / 2.0);
      const State k4 = liouville_operator(state + k3) * dt;

      return state + k1 * (1.0 / 6.0) + k2 * (1.0 / 3.0) + k3 * (1.0 / 3.0) +
             k4 * (1.0 / 6.0);
    };
  }
};

} // namespace dvr_smd
}

#endif //METHODS_DVR_SMD_H
