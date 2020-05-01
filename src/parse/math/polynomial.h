#ifndef PARSE_MATH_POLYNOMIAL_H
#define PARSE_MATH_POLYNOMIAL_H

#include <quartz>
#include <boost/property_tree/ptree.hpp>

namespace quartz {
namespace parse {

namespace ptree = boost::property_tree;

math::Polynomial<double> polynomial(const ptree::ptree & input);

template<typename T>
math::Polynomial<T> polynomial(const MathObject<T> & mathobject) {
  if(mathobject.type != math::OperatorType::Function) {
    throw Error("The math object is not a polynomial");
  }

  const auto var = mathobject.value;

  if(std::holds_alternative<math::Polynomial<T>>(var->value)) {
    return std::get<math::Polynomial<T>>(var->value);
  } else {
    throw Error("The math object is not a polynomial");
  }
}

}
}

#endif //PARSE_MATH_POLYNOMIAL_H
