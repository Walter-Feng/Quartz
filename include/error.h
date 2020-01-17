#ifndef QUARTZ_ERROR_H
#define QUARTZ_ERROR_H

#include <stdexcept>
#include <string>

namespace quartz {

struct Error : public std::runtime_error {
  explicit Error (const std::string & error_message) :
    std::runtime_error("Error: " + error_message) {}
  explicit Error (const char * error_message) :
    std::runtime_error("Error: " + std::string(error_message)) {}
};

}

#endif //QUARTZ_ERROR_H
