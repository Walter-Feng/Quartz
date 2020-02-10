#ifndef UTIL_MEMBER_FUNCTION_WRAPPER_H
#define UTIL_MEMBER_FUNCTION_WRAPPER_H

#include "check_member.h"

template<typename Function, typename T>
auto at(const Function & function, const arma::Mat<T> & positions) {

  const bool is_valid = has_at<Function, double(const arma::vec &)>::value ||
                        has_at<Function, cx_double(const arma::vec &)>::value;

  if(!is_valid) {
    throw Error("the function passed in has no at member function");
  }

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

#endif //UTIL_MEMBER_FUNCTION_WRAPPER_H
