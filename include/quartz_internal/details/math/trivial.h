#ifndef TRIVIAL_TRIVIAL_H
#define TRIVIAL_TRIVIAL_H

namespace math {

inline
double factorial(const double n) {
  return std::tgamma(n + 1);
}

}

#endif //TRIVIAL_TRIVIAL_H
