#ifndef PARSE_MATH_POLYNOMIAL_H
#define PARSE_MATH_POLYNOMIAL_H

#include <quartz>
#include <boost/property_tree/ptree.hpp>

namespace quartz {
namespace parse {

namespace ptree = boost::property_tree;

math::Polynomial<double> polynomial(const ptree::ptree & input);

}
}

#endif //PARSE_MATH_POLYNOMIAL_H
