#ifndef UTIL_AUTO_GENERATOR_H
#define UTIL_AUTO_GENERATOR_H

#include "quartz_internal/details/math/polynomial.h"

template<typename T>
math::Polynomial<T>
kinetic_energy(const arma::uword real_space_dim, const arma::vec & masses) {
  if (masses.n_elem != real_space_dim) {
    throw Error("the masses provided does not match the dimension");
  }

  return math::Polynomial<T>(0.5 * arma::ones(real_space_dim) / masses,
                             arma::join_cols(arma::zeros<lvec>(real_space_dim,
                                                               real_space_dim),
                                             2 * arma::eye<lvec>(real_space_dim,
                                                                 real_space_dim)));
}

template<typename T>
math::Polynomial<T> kinetic_energy(const arma::uword real_space_dim) {
  return kinetic_energy<T>(real_space_dim, arma::ones(real_space_dim));
}

template<typename T>
math::Polynomial<T>
hamiltonian(const math::Polynomial<T> & potential, const arma::vec & masses) {
  const auto dim = potential.dim();

  const auto potential_part =
      math::Polynomial<T>(potential.coefs,
                          arma::join_cols(potential.exponents,
                                          arma::zeros<lmat>(dim,
                                                            potential.exponents.n_cols)));

  return potential_part + kinetic_energy<T>(dim, masses);
}

template<typename T>
math::Polynomial<T> hamiltonian(const math::Polynomial<T> & potential) {
  return hamiltonian<T>(potential, arma::ones(potential.dim()));
}

inline
std::vector<math::Polynomial<double>> polynomial_observables(
    const arma::uword dim,
    const arma::uword grade) {

  std::vector<math::Polynomial<double>> op(std::pow(grade, dim));
  const arma::uvec grid = grade * arma::ones<arma::uvec>(dim);
  const arma::uvec table = math::space::grids_to_table(grid);

  op[0] = math::Polynomial<double>(dim, 1.0);

  for (arma::uword i = 1; i < op.size(); i++) {
    op[i] =
        math::Polynomial(math::polynomial::Term<double>(1.0,
                                                        math::space::index_to_indices(
                                                            i,
                                                            table)));
  }

  return op;
}



#endif //UTIL_AUTO_GENERATOR_H
