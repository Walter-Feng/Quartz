#ifndef PARSE_GAUSSIAN_H
#define PARSE_GAUSSIAN_H

#include <quartz>
#include <boost/property_tree/ptree.hpp>

namespace quartz {
namespace parse {

namespace ptree = boost::property_tree;

math::Gaussian<cx_double> gaussian(const ptree::ptree & input);

}
}


#endif //PARSE_GAUSSIAN_H
