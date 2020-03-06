#ifndef METHODS_DMD_H
#define METHODS_DMD_H

namespace method {
// use your method name to create a subspace for your
// implementation of details
namespace dmd {

namespace details {

template<typename Potential>
arma::mat effective_force(const Potential & potential,
                          const arma::mat & points,
                          const arma::vec & masses,
                          const double beta) {
  arma::mat result = arma::mat(points.n_rows / 2, points.n_cols);

#pragma omp parallel for
  for (arma::uword i = 0; i < points.n_cols; i++) {
    const arma::vec point = points.col(i);
    const arma::vec position = points.col(i).rows(0, points.n_rows / 2 - 1);
    for (arma::uword j = 0; j < points.n_rows / 2; j++) {
      const auto p_j = j + points.n_rows / 2;
      math::Polynomial<double> correction_term =
          math::Polynomial<double>(points.n_rows,std::pow(beta / masses(j), 2));
      correction_term.indices(p_j,0) = 2;

      result(j, i) = -potential.derivative(j).at(position)
                     + potential.derivative(j).derivative(j).derivative(j).at(
          position)
                       * (correction_term - beta / masses(j)).at(
                           point) / 24.0;
    }
  }

  return result;
}

}

using State = md::State;


template<typename Potential>
struct Operator {

private:
  PropagationType type = Classic;

public:
  Potential potential;
  double beta;

  Operator(const State & state,
           const Potential & potential,
           const double beta = 1.0) :
      potential(potential),
      beta(beta){}


  inline
  PropagationType propagation_type() const {
    return Classic;
  }

  State operator()(const State & state) const {

    arma::mat p_submatrix = state.points.rows(state.dim(), 2 * state.dim() - 1);
    p_submatrix.each_col() /= state.masses;

    const arma::mat change_list =
        arma::join_cols(p_submatrix,
                        details::effective_force(this->potential,
                                                 state.points,
                                                 state.masses,
                                                 this->beta));

    return State(change_list, state.weights, state.masses);
  }

};
}
}
#endif //METHODS_DMD_H
