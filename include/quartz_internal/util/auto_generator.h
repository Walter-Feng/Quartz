#ifndef UTIL_AUTO_GENERATOR_H
#define UTIL_AUTO_GENERATOR_H

#include "quartz_internal/details/math/polynomial.h"

template<typename T>
math::Polynomial<T> kinetic_energy(const arma::uword real_space_dim, const arma::vec & masses) {
  if(masses.n_elem != real_space_dim) {
    throw Error("the masses provided does not match the dimension");
  }

  return math::Polynomial<T>(0.5 * arma::ones(real_space_dim) / masses,
                             arma::join_cols(arma::zeros<lvec>(real_space_dim,real_space_dim),
                                             2 * arma::eye<lvec>(real_space_dim,real_space_dim)));
}

template<typename T>
math::Polynomial<T> kinetic_energy(const arma::uword real_space_dim) {
  return kinetic_energy<T>(real_space_dim, arma::ones(real_space_dim));
}

template<typename T>
math::Polynomial<T> hamiltonian(const math::Polynomial<T> & potential, const arma::vec & masses) {
  const auto dim = potential.dim();

  const auto potential_part =
      math::Polynomial<T>(potential.coefs,
                          arma::join_cols(potential.indices,
                                          arma::zeros<lvec>(dim,dim)));

  return potential_part + kinetic_energy<T>(dim, masses);
}

template<typename T>
math::Polynomial<T> hamiltonian(const math::Polynomial<T> & potential) {
  return hamiltonian<T>(potential, arma::ones(potential.dim()));
}


#endif //UTIL_AUTO_GENERATOR_H