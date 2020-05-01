#include "math_object.h"

#include "exponential.h"
#include "gaussian.h"
#include "polynomial.h"

#include "src/util/ptree.h"

namespace quartz {
namespace parse {

namespace ptree = boost::property_tree;

ElementaryFunction<double> elementary_function(const ptree::ptree & input) {

  if(input.get_child_optional("exponential")) {
    return {exponential(input.get_child("exponential"))};
  } else if(input.get_child_optional("polynomial")) {
    return {polynomial(input.get_child("polynomial"))};
  } else if(input.get_child_optional("gaussian")) {
    return {math::GaussianWithPoly<double>(gaussian(input.get_child("gaussian")).abs())};
  } else {
    throw Error("Function does not exist");
  }
}

MathObject<double> math_object(const ptree::ptree & input) {
  if(input.get_child_optional("exponential")) {

    return MathObject<double>({exponential(input.get_child("exponential"))});

  } else if(input.get_child_optional("polynomial")) {

    return MathObject<double>({polynomial(input.get_child("polynomial"))});

  } else if(input.get_child_optional("gaussian")) {

    return MathObject<double>({math::GaussianWithPoly<double>(gaussian(input.get_child("gaussian")).abs())});

  } else if(input.get_child_optional("sum")) {

    return math_object(input.get_child("sum")) + math_object(input.get_child("sum.with"));

  } else if(input.get_child_optional("subtract")) {

    return math_object(input.get_child("subtract")) - math_object(input.get_child("subtract.with"));

  } else if(input.get_child_optional("multiply")) {

    return math_object(input.get_child("multiply")) * math_object(input.get_child("multiply.with"));

  } else if(input.get_child_optional("divide")) {

    return math_object(input.get_child("divide")) / math_object(input.get_child("divide.with"));

  } else {
    throw Error("Function does not exist");
  }
}

}
}