#ifndef TRIVIAL_TRIVIAL_H
#define TRIVIAL_TRIVIAL_H

namespace math {

inline
double factorial(const double n) {
  if (n == 0) return 1;
  else if (n == 1) return n;
  else return n * factorial(n - 1);
}

}

#endif //TRIVIAL_TRIVIAL_H
