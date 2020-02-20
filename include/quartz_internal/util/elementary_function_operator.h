#ifndef UTIL_ELEMENTARY_FUNCTION_OPERATOR_H
#define UTIL_ELEMENTARY_FUNCTION_OPERATOR_H

template<typename Functor, typename Arg>
auto exp(const Functor & functor, const Arg & arg, const arma::uword cut_off) {

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


#endif //UTIL_ELEMENTARY_FUNCTION_OPERATOR_H
