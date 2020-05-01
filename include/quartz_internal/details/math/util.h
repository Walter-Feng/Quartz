#ifndef MATH_UTIL_H
#define MATH_UTIL_H

namespace math {
namespace util {

inline
std::vector<std::string> variable_names(const arma::uword dim) {

  std::vector<std::string> variables;

  if (dim == 1) {
    variables = {"x"};
  } else if (dim == 2) {
    variables = {"x", "y"};
  } else if (dim == 3) {
    variables = {"x", "y", "z"};
  } else {
    variables = {};
    for (arma::uword i = 0; i < dim; i++) {
      variables.emplace_back("x" + std::to_string(i));
    }
  }

  return variables;

}

template<typename T>
arma::Mat<T> direct_sum(const arma::Cube<T> & mats) {
  if (mats.n_cols != mats.n_rows) {
    throw Error("direct_sum: the matrices are not squared");
  }

  arma::Mat<T> result(mats.n_rows * mats.n_slices,
                      mats.n_cols * mats.n_slices,
                      arma::fill::zeros);

#pragma omp parallel for
  for(arma::uword i=0; i<mats.n_slices; i++) {
    result(arma::span(i * mats.n_rows, (i+1) * mats.n_rows - 1),
           arma::span(i * mats.n_cols, (i+1) * mats.n_cols - 1)) = mats.slice(i);
  }

  return result;
}

}
}

#endif //MATH_UTIL_H
