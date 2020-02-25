#ifndef UTIL_AUTO_GENERATOR_H
#define UTIL_AUTO_GENERATOR_H

#include "quartz_internal/details/math/polynomial.h"

template<typename T>
math::Polynomial<T> kinetic_energy(const arma::uword real_space_dim) {
  return math::Polynomial<T>(0.5 * arma::ones(real_space_dim),
                             arma::join_cols(arma::zeros<lvec>(real_space_dim,real_space_dim),
                                             2 * arma::eye<lvec>(real_space_dim,real_space_dim)));
}

template<typename T>
math::Polynomial<T> hamiltonian(const math::Polynomial<T> & potential) {
  const auto dim = potential.dim();

  const auto potential_part =
      math::Polynomial<T>(potential.coefs,
                          arma::join_cols(potential.indices,
                                          arma::zeros<lvec>(dim,dim)));

  return potential_part + kinetic_energy<T>(dim);
}


#endif //UTIL_AUTO_GENERATOR_H
