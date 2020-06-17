#ifndef QUARTZ_EXPONENTIAL_H
#define QUARTZ_EXPONENTIAL_H

#include <quartz>

#include "src/util/ptree.h"

namespace quartz {
namespace parse {

namespace ptree = boost::property_tree;

math::Exponential<double> exponential(const ptree::ptree & input);

}
}

#endif //QUARTZ_EXPONENTIAL_H
