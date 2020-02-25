#ifndef UTIL_MEMBER_FUNCTION_WRAPPER_H
#define UTIL_MEMBER_FUNCTION_WRAPPER_H

#include "check_member.h"

template<typename Function, typename T>
auto at(const Function & function, const arma::Mat<T> & positions) {

//TODO: It is pretty weird that the following SFINAE does not work
//  const bool is_valid = has_at<Function, double(const arma::Mat<T> &)>::value ||
//                        has_at<Function, cx_double(const arma::Mat<T> &)>::value;
//  if(!is_valid) {
//    throw Error("the function passed in has no at member function");
//  }

  const arma::vec initial = positions.col(0);
  auto result =
      arma::Col<decltype(function.at(initial))>(positions.n_cols);
  #pragma omp parallel for
  for(arma::uword i=0; i<positions.n_cols; i++) {
    const arma::vec point_i = positions.col(i);
    result(i) = function.at(point_i);
  }

  return result;
}

template<typename Function>
auto derivative(const Function & function, const arma::uword index) {
  return function.derivative(index);
}

template<typename Function>
auto derivative(const Function & function, const arma::uvec & index) {

  Function result = function;

  for (arma::uword i = 0; i < index.n_elem; i++) {
    for (arma::uword j = 0; j < index(i); j++) {
      result = result.derivative(i);
    }
  }

  return result;
}


#endif //UTIL_MEMBER_FUNCTION_WRAPPER_H
