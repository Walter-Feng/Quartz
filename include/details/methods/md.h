#ifndef QUARTZ_MD_H
#define QUARTZ_MD_H

namespace method {
// use your method name to create a subspace for your
// implementation of details
namespace md {

namespace details {

template<typename Potential>
arma::vec force(const Potential & potential,
                const arma::vec & position) {

//TODO(Rui): It seems that the potential function does not need to return
// potential struct itself for its derivatives.
//  static_assert(has_derivative<Potential, Potential(arma::uword)>::value,
//                "The potential provided does not allow derivative, "
//                "thus unable to calculate force.");

  arma::vec result = arma::vec(position.n_elem);

#pragma omp parallel for
  for (arma::uword i = 0; i < position.n_elem; i++) {
    result(i) = potential.derivative(i).at(position);
  }

  return result;
}

template<typename Potential>
arma::mat force(const Potential & potential,
                const arma::mat & positions) {
  arma::mat result = arma::mat(arma::size(positions));

#pragma omp parallel for
  for (arma::uword i = 0; i < positions.n_cols; i++) {
    result.col(i) = force(potential, positions.col(i));
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
      weights(at(initial, points)),
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
      weights(at(initial, points)),
      masses(arma::ones<arma::vec>(arma::prod(grid))) {
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
  arma::vec positional_expectation() const {

    arma::uword dim = this->dim();

    arma::mat points_copy_positional_part = this->points.rows(0, dim - 1);
    points_copy_positional_part.each_row() %= weights.t();

    return arma::sum(points_copy_positional_part, 1);
  }

  inline
  arma::vec momentum_expectation() const {
    arma::uword dim = this->dim();

    arma::mat points_copy_momentum_part = this->points.rows(dim, 2 * dim - 1);
    points_copy_momentum_part.each_row() %= weights.t();

    return arma::sum(points_copy_momentum_part, 1);
  }
};


struct Operator {

private:
  PropagationType type = Classic;

public:
  arma::mat change_list;

  template<typename Potential>
  Operator(const State & state,
           const Potential & potential) :
      change_list(
          arma::join_cols(state.points.rows(state.dim(), 2 * state.dim() - 1),
                          details::force(potential, state.points))) {}

  explicit
  Operator(const arma::mat & change_list) :
      change_list(change_list) {}


  inline
  PropagationType propagation_type() const {
    return Classic;
  }

  State operator*(const State & state) const {

    if (this->change_list.n_rows != state.points.n_rows ||
        this->change_list.n_cols != state.points.n_cols) {
      throw Error("The operator does not match the state");
    }

    return State(this->change_list, state.weights, state.masses);
  }

  Operator operator+(const Operator & B) const {
    const arma::mat new_change = this->change_list + B.change_list;
    return Operator(new_change);
  }

  Operator operator-(const Operator & B) const {
    const arma::mat new_change = this->change_list - B.change_list;
    return Operator(new_change);
  }

  template<typename T>
  Operator operator*(const T & B) const {
    const arma::mat new_change = this->change_list * B;
    return Operator(new_change);
  }

};

} // namespace md
}


#endif //QUARTZ_MD_H
