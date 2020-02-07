#ifndef QUARTZ_TYPE_CONVERTER_H
#define QUARTZ_TYPE_CONVERTER_H

#include "error.h"

namespace utils {

template<typename T>
T convert(const double number) {
  return number;
}

template<>
inline
arma::cx_double convert(const double number) {
  return {number,0.0};
}

}

#endif //QUARTZ_TYPE_CONVERTER_H
