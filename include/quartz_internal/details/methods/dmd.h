#ifndef METHODS_DMD_H
#define METHODS_DMD_H

namespace method {
// use your method name to create a subspace for your
// implementation of details
namespace dmd {

namespace details {

template<typename Initial, typename Potential>
arma::mat effective_force(const Potential & potential,
                          const Initial & initial,
                          const arma::mat & points) {
  arma::mat result = arma::mat(points.n_rows / 2, points.n_cols);

#pragma omp parallel for
  for (arma::uword i = 0; i < points.n_cols; i++) {
    const arma::vec point = points.col(i);
    const arma::vec position = points.col(i).rows(0, points.n_rows / 2 - 1);
    for (arma::uword j = 0; j < points.n_rows / 2; j++) {
      const auto p_j = j + points.n_rows / 2;
      result(j, i) = -potential.derivative(j).at(position)
                     + potential.derivative(j).derivative(j).derivative(j).at(
          position)
                       * initial.derivative(p_j).derivative(p_j).at(point)
                       / initial.at(point) / 24.0;
    }
  }

  return result;
}

}

using State = md::State;


template<typename Potential, typename Initial>
struct Operator {

private:
  PropagationType type = Classic;

public:
  Potential potential;
  Initial initial;

  Operator(const State & state,
           const Initial & initial,
           const Potential & potential) :
      potential(potential),
      initial(initial) {}


  inline
  PropagationType propagation_type() const {
    return Classic;
  }

  State operator()(const State & state) const {

    arma::mat p_submatrix = state.points.rows(state.dim(), 2 * state.dim() - 1);
    p_submatrix.each_col() /= state.masses;

    const arma::mat change_list =
        arma::join_cols(p_submatrix,
                        details::effective_force(potential,
                                                 this->initial,
                                                 state.points));

    return State(change_list, state.weights, state.masses);
  }

};
}
}
#endif //METHODS_DMD_H
