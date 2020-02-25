#ifndef MATH_MOYAL_BRACKET_H
#define MATH_MOYAL_BRACKET_H

#include "space.h"

#include "quartz_internal/error.h"
#include "quartz_internal/util/elementary_function_operator.h"

namespace math {

namespace details {

inline
Polynomial<double> polynomial_poisson_operator(arma::uword n_real_space_dim) {

  Polynomial<double> poisson_operator(n_real_space_dim * 4);

  arma::uvec table = space::grids_to_table(arma::uvec{n_real_space_dim, 2, 2});

  Polynomial<double> result(n_real_space_dim * 4);

  for (arma::uword i = 0; i < n_real_space_dim; i++) {
    // creates arma::uvec of dimension [n_real_space_dim, 2, 2], where the first 2 represents
    // the x and corresponding p, while the second 2 represents the left function
    // (or the first function argument to be derived) and the other function (usually the
    // Hamiltonian function)

    auto xp = lvec(n_real_space_dim * 2 * 2, arma::fill::zeros);
    auto px = lvec(n_real_space_dim * 2 * 2, arma::fill::zeros);

    xp(space::indices_to_index(arma::uvec{i, 0, 0},
                               table)) = 1; // set the d_x operator to the left function
    xp(space::indices_to_index(arma::uvec{i, 1, 1},
                               table)) = 1; // set the d_p operator to the right function
    px(space::indices_to_index(arma::uvec{i, 1, 0},
                               table)) = 1; // set the d_p operator to the left function
    px(space::indices_to_index(arma::uvec{i, 0, 1},
                               table)) = 1; // set the d_x operator to the right function

    result = result + polynomial::Term<double>(1, xp) +
             polynomial::Term<double>(-1, xp);
  }

  return result;

}

template<typename A, typename H>
auto poisson_operate(const Polynomial<double> & poisson_operator,
                     const A & a,
                     const H & h) {

  if (poisson_operator.dim() % 4 != 0) {
    throw Error("The polynomial provided is not likely a poisson operator");
  }

  const arma::uword dim = poisson_operator.dim() / 4;

  const auto differentiate = [&dim](const polynomial::Term<double> & term,
                                    const A & a,
                                    const H & h) {
    return
        polynomial::Term<double>(term.coef, term.indices(
            arma::span(0, dim * 2 - 1))).differentiate(a)
        *
        polynomial::Term<double>(1.0, term.indices(
            arma::span(dim * 2, dim * 4 - 1))).differentiate(h);
  };

  auto result = differentiate(poisson_operator.term(0), a, h);
  for (arma::uword i = 1; i < poisson_operator.coefs.n_elem; i++) {
    result = result + differentiate(poisson_operator.term(i), a, h);
  }

  return result;
}
}

template<typename A, typename H>
auto moyal_bracket(const A & a, const H & h, const arma::uword cut_off) {

  if (a.dim() != h.dim()) {
    throw Error("Different dimension between the operator and the hamiltonian");
  }

  if (a.dim() % 2 != 0) {
    throw Error("The operator provided is not likely in the phase space form");
  }

  const auto dim = a.dim() / 2;

  const auto poisson_op = details::polynomial_poisson_operator(dim);

  auto result = details::poisson_operate(poisson_op, a, h);
  for (arma::uword i = 1; i < cut_off; i++) {
    result = result +
        details::poisson_operate(
            std::pow(-1, i) / factorial(2 * i + 1) / std::pow(2, 2*i)
            * poisson_op.pow(2 * i + 1),
            a, h);
  }

  return result;
}

template<typename A, typename Functor, typename H>
auto moyal_bracket(const Functor & post_functor,
                   const A & a,
                   const H & h,
                   const arma::uword cut_off) {

  if (a.dim() != h.dim()) {
    throw Error("Different dimension between the operator and the hamiltonian");
  }

  if (a.dim() % 2 != 0) {
    throw Error("The operator provided is not likely in the phase space form");
  }

  const auto dim = a.dim() / 2;

  const auto poisson_op = details::polynomial_poisson_operator(dim);

  auto result = details::poisson_operate(poisson_op, a, h);
  for (arma::uword i = 1; i < cut_off; i++) {
    result = result +
             details::poisson_operate(
                 std::pow(-1, i) / factorial(2 * i + 1) / std::pow(2, 2*i)
                 * poisson_op.pow(2 * i + 1),
                 a, h);
  }

  return result;
}

}

#endif //MATH_MOYAL_BRACKET_H
