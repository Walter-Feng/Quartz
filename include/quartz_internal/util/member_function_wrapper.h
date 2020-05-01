#ifndef UTIL_MEMBER_FUNCTION_WRAPPER_H
#define UTIL_MEMBER_FUNCTION_WRAPPER_H

#include "check_member.h"

template<typename T>
std::common_type_t<T, double> at(const double function,
                                 const arma::Col<T> & position) {
  return std::common_type_t<T, double>{function};
}

template<typename T>
cx_double at(const cx_double function,
             const arma::Col<T> & position) {
  return function;
}

template<typename Function, typename T>
auto at(const Function & function, const arma::Col<T> & position) {
  return function.at(position);
}

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
  for (arma::uword i = 0; i < positions.n_cols; i++) {
    const arma::vec point_i = positions.col(i);
    result(i) = function.at(point_i);
  }

  return result;
}

inline
double derivative(const double function, const arma::uword index) {
  return 0.0;
}

inline
cx_double derivative(const cx_double function, const arma::uword index) {
  return cx_double{0.0};
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

template<typename Operator, typename State, typename Potential>
OperatorWrapper<Operator, State, Potential>
    normalise = [](const Operator &,
                   const Potential &) -> Propagator<State> {
  return [](const State & state, const double dt) -> State {
    return state.normalise();
  };
};


#endif //UTIL_MEMBER_FUNCTION_WRAPPER_H
