#ifndef QUARTZ_MATH_OBJECT_H
#define QUARTZ_MATH_OBJECT_H

#include <quartz>
#include <boost/property_tree/ptree.hpp>

namespace quartz {
namespace parse {

namespace ptree = boost::property_tree;

ElementaryFunction<double> elementary_function(const ptree::ptree & input);

MathObject<double> math_object(const ptree::ptree & input);

}
}
#endif //QUARTZ_MATH_OBJECT_H
