#ifndef QUARTZ_ERROR_H
#define QUARTZ_ERROR_H

#include <stdexcept>
#include <string>
#include <cassert>


struct Error : public std::runtime_error {
  explicit Error (const std::string & error_message) :
    std::runtime_error("Error: " + error_message) {}
  explicit Error (const char * error_message) :
    std::runtime_error("Error: " + std::string(error_message)) {}
};

inline
void Warning(const std::string & warning_message) {
  std::cout << "Warning: " + warning_message;
}


#endif //QUARTZ_ERROR_H
