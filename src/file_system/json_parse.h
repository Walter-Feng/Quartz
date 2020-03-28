#ifndef EXE_FILE_SYTSEM_JSON_PARSE_H
#define EXE_FILE_SYTSEM_JSON_PARSE_H

#include <boost/property_tree/ptree.hpp>

namespace quartz {
namespace file_system {

namespace ptree = boost::property_tree;

ptree::ptree parse(const std::string & file);

}
}

#endif //EXE_FILE_SYTSEM_JSON_PARSE_H
