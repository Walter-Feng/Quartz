#ifndef UTIL_ELEMENTARY_FUNCTION_OPERATOR_H
#define UTIL_ELEMENTARY_FUNCTION_OPERATOR_H

#include "quartz_internal/details/math/trivial.h"

template<typename Functor, typename Arg>
auto exp(const Functor & functor, const Arg & arg, const arma::uword cut_off) {

  auto result = functor(arg);

  for (arma::uword i = 2; i < cut_off; i++) {
    auto term_result = functor(arg);
    for (arma::uword j = 1; j < i; j++) {
      term_result = functor(term_result);
    }

    result = result + term_result / math::factorial(i);
  }
  return result + arg;
}

template<typename Functor, typename PostFunctor, typename Arg>
auto exp(const Functor & functor,
         const PostFunctor & post_functor,
         const Arg & arg,
         const arma::uword cut_off) {

  auto result = post_functor(functor(arg));

  for (arma::uword i = 2; i < cut_off; i++) {
    auto term_result = functor(arg);
    for (arma::uword j = 1; j < i; j++) {
      term_result = functor(term_result);
    }

    result = result + post_functor(term_result) / math::factorial(i);
  }
  return result + post_functor(arg);
}

template<typename Functor, typename Arg>
auto sin(const Functor & functor, const Arg & arg, const arma::uword cut_off) {

  const std::function<double(double)>
      factorial = [&factorial](const double n) -> double {
    if (n == 0) return 1;
    else if (n == 1) return n;
    else return n * factorial(n - 1);
  };

  auto result = functor(arg);

  for (arma::uword i = 2; i < cut_off; i++) {
    auto term_result = functor(arg);
    for (arma::uword j = 1; j < i; j++) {
      term_result = functor(term_result);
    }

    result = result + term_result / (double) factorial(i);
  }
  return result + arg;
}

template<typename Functor, typename PostFunctor, typename Arg>
auto sin(const Functor & functor,
         const PostFunctor & post_functor,
         const Arg & arg,
         const arma::uword cut_off) {

  const std::function<double(double)>
      factorial = [&factorial](const double n) -> double {
    if (n == 0) return 1;
    else if (n == 1) return n;
    else return n * factorial(n - 1);
  };

  auto result = post_functor(functor(arg));

  for (arma::uword i = 2; i < cut_off; i++) {
    auto term_result = functor(arg);
    for (arma::uword j = 1; j < i; j++) {
      term_result = functor(term_result);
    }

    result = result + post_functor(term_result) / (double) factorial(i);
  }
  return result + post_functor(arg);
}


#endif //UTIL_ELEMENTARY_FUNCTION_OPERATOR_H
