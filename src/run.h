#ifndef QUARTZ_RUN_H
#define QUARTZ_RUN_H

#include <boost/property_tree/ptree.hpp>

namespace quartz {

namespace ptree = boost::property_tree;

int run(const ptree::ptree & input);


}

#endif //QUARTZ_RUN_H
