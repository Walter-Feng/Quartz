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
  arma::vec result = arma::vec(position.n_elem);

#pragma omp parallel for
  for(arma::uword i=0; i<position.n_elem; i++) {
    result(i) = potential.derivative(i).at(position);
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
  arma::uword dim() {
    return points.n_rows / 2;
  }

  inline
  arma::vec positional_expectation() {

    arma::uword dim = this->dim();

    arma::mat points_copy_positional_part = this->points.rows(0,dim-1);
    points_copy_positional_part.each_row() %= weights.t();

    return arma::sum(points_copy_positional_part,1);
  }

  inline
  arma::vec momentum_expectation() {
    arma::uword dim = this->dim();

    arma::mat points_copy_momentum_part = this->points.rows(dim,2*dim-1);
    points_copy_momentum_part.each_row() %= weights.t();

    return arma::sum(points_copy_momentum_part,1);
  }
};

template<typename Potential>
State propagator(State state,
                 const Potential & potential,
                 const double dt) {
  // only need the potential defined over the real space
  // requiring the potential to have .at() and .derivative() as a member function

  arma::mat new_points = arma::mat(arma::size(state.points));

#pragma omp parallel for
  for(arma::uword i=0;i<new_points.n_cols;i++) {
    new_points.col(i) = state.points.col(i);
  }

  state.points = new_points;
  return state;
}

} // namespace md
}


#endif //QUARTZ_MD_H
