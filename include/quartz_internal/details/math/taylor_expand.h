#ifndef MATH_TAYLOR_EXPAND_H
#define MATH_TAYLOR_EXPAND_H

namespace math {
namespace details {

inline
arma::umat indices_with_same_sum(const arma::uword dim,
                                 const arma::uword sum) {

  const arma::umat all =
      math::space::auto_iteration_over_dims(
          (sum + 1) * arma::ones<arma::uvec>(dim));
  const arma::uvec each_sum = arma::sum(all).t();

  return all.cols(arma::find(each_sum == sum));
}
}


template<typename Function, typename T>
auto taylor_expand(const Function & function,
                   const arma::uword grade) {

  std::vector<std::vector<Function>> function_derivative_map(grade);
  std::vector<arma::umat> indices(grade);

  for (arma::uword i = 0; i < grade; i++) {
    const arma::uword true_i = i + 1;
    const arma::umat indices_at_true_i =
        details::indices_with_same_sum(function.dim(), true_i);

    indices[i] = indices_at_true_i;

    std::vector<Function> function_derivatives_at_i(indices_at_true_i.n_cols);

#pragma omp parallel for
    for (arma::uword j = 0; j < indices_at_true_i.n_cols; j++) {
      const arma::uvec derivative_operator =
          indices_at_true_i.col(j);
      function_derivatives_at_i[j] = derivative(function, derivative_operator);
    }

    function_derivative_map[i] = function_derivatives_at_i;
  }

  return [indices, function_derivative_map, function, grade](
      const arma::Col<T> & position,
      const arma::Col<T> & translation) -> auto {

    auto result = function.at(translation);

    for (arma::uword i = 0; i < grade; i++) {
#pragma omp parallel for
      for (arma::uword j = 0; j < indices[i].size(); j++) {
        const math::polynomial::Term<T> term(T{1.0}, indices[i].col(j));

        const arma::Col<T> tranlated_position = position - translation;
        result += function_derivative_map[i][j].at(translation) *
                  term.at(tranlated_position) / factorial(i + 1);
      }
    }

    return result;
  };

}

template<typename Function, typename T>
auto taylor_expand(const Function & function,
                   const arma::vec & translation,
                   const arma::uword grade) {

  math::Polynomial<T> result(function.dim(), function.at(translation));

  for (arma::uword i = 0; i < grade; i++) {
    const arma::uword true_i = i + 1;
    const arma::umat indices_at_true_i =
        details::indices_with_same_sum(function.dim(), true_i);

#pragma omp parallel for
    for (arma::uword j = 0; j < indices_at_true_i.n_cols; j++) {
      const arma::uvec derivative_operator =
          indices_at_true_i.col(j);

      const math::polynomial::Term<T>
          term(derivative(function, derivative_operator).at(translation)
               / factorial(i + 1), derivative_operator);

      result = result + term;
    }
  }

  return result;
}

}

#endif //MATH_TAYLOR_EXPAND_H
