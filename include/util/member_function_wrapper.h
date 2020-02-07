#ifndef UTIL_MEMBER_FUNCTION_WRAPPER_H
#define UTIL_MEMBER_FUNCTION_WRAPPER_H

template<typename Function, typename T>
auto at(const Function & function, const arma::Mat<T> positions) {
  auto result =
      arma::Col<decltype(function.at(positions.col(0)))>(positions.n_cols);
  #pragma omp parallel for
  for(arma::uword i=0; i<positions.n_cols; i++) {
    result(i) = function.at(positions.col(i));
  }

  return result;
}

#endif //UTIL_MEMBER_FUNCTION_WRAPPER_H
