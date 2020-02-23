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

  auto result = functor(arg);

  for (arma::uword i = 3; i < cut_off; i += 2) {
    auto term_result = functor(arg);
    for (arma::uword j = 1; j < i; j++) {
      term_result = functor(term_result);
    }

    result =
        result + term_result * std::pow(-1, (i - 1) / 2) / math::factorial(i);
  }
  return result;
}

template<typename Functor, typename PostFunctor, typename Arg>
auto sin(const Functor & functor,
         const PostFunctor & post_functor,
         const Arg & arg,
         const arma::uword cut_off) {
  auto result = post_functor(functor(arg));

  for (arma::uword i = 3; i < cut_off; i += 2) {
    auto term_result = functor(arg);
    for (arma::uword j = 1; j < i; j++) {
      term_result = functor(term_result);
    }

    result = result + post_functor(term_result) * std::pow(-1, (i - 1) / 2) /
                      math::factorial(i);
  }
  return result;
}

template<typename Functor, typename Arg>
auto cos(const Functor & functor, const Arg & arg, const arma::uword cut_off) {

  auto result = functor(functor(arg)) / (-2.0);

  for (arma::uword i = 4; i < cut_off; i += 2) {
    auto term_result = functor(arg);
    for (arma::uword j = 1; j < i; j++) {
      term_result = functor(term_result);
    }

    result = result + term_result * std::pow(-1, i / 2) / math::factorial(i);
  }
  return arg + result;
}

template<typename Functor, typename PostFunctor, typename Arg>
auto cos(const Functor & functor,
         const PostFunctor & post_functor,
         const Arg & arg,
         const arma::uword cut_off) {

  auto result = post_functor(functor(functor(arg))) / (-2.0);

  for (arma::uword i = 4; i < cut_off; i += 2) {
    auto term_result = functor(arg);
    for (arma::uword j = 1; j < i; j++) {
      term_result = functor(term_result);
    }

    result = result + post_functor(term_result) * std::pow(-1, i / 2) /
                      math::factorial(i);
  }
  return post_functor(arg) + result;
}


#endif //UTIL_ELEMENTARY_FUNCTION_OPERATOR_H
