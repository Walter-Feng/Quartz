#ifndef QUARTZ_TYPE_CONVERTER_H
#define QUARTZ_TYPE_CONVERTER_H

#include "quartz_internal/error.h"

namespace utils {

template<typename T>
T convert(const double number) {
  return number;
}

template<>
inline
arma::cx_double convert(const double number) {
  return {number, 0.0};
}

template<typename T>
std::string to_string_with_precision(const T a_value,
                                     const int n = 6,
                                     const int width = 12) {
  std::ostringstream out;
  out.precision(n);
  out.width(width);
  out << std::fixed << a_value;
  return out.str();
}

}

#endif //QUARTZ_TYPE_CONVERTER_H
