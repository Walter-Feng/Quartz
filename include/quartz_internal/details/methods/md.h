#ifndef METHODS_MD_H
#define METHODS_MD_H

namespace method {
// use your method name to create a subspace for your
// implementation of details
namespace md {

namespace details {

template<typename Potential>
arma::mat force(const Potential & potential,
                const arma::mat & positions) {
  arma::mat result = arma::mat(arma::size(positions));

#pragma omp parallel for
  for (arma::uword i = 0; i < positions.n_cols; i++) {
    const arma::vec position = positions.col(i);
    for (arma::uword j = 0; j < positions.n_rows; j++) {
      result(j,i) = - potential.derivative(j).at(position);
    }
  }

  return result;
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
  double expectation(const Function & function) const {
    auto result = at(function, this->points);

    return arma::dot(result, weights) / arma::sum(weights);
  }

  inline
  arma::vec positional_expectation() const {
    arma::uword dim = this->dim();

    return this->points.rows(0, dim - 1) * this->weights / arma::sum(this->weights);
  }

  inline
  arma::vec momentum_expectation() const {
    arma::uword dim = this->dim();

    return this->points.rows(dim, 2 * dim -1) * this->weights / arma::sum(this->weights);
  }

  State operator+(const State & B) const {
    if (!arma::approx_equal(this->weights, B.weights, "abs_diff", 1e-16) ||
        !arma::approx_equal(this->masses, B.masses, "abs_diff", 1e-16)) {
      throw Error("Different md states are being added");
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

} // namespace md
}


#endif //METHODS_MD_H
