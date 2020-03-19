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

}
}

#endif //MATH_UTIL_H
