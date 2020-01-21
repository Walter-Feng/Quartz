#ifndef MATH_SINUSOIDAL_H
#define MATH_SINUSOIDAL_H

namespace quartz {
namespace math {

template<typename T>
struct Sinusoidal {
  arma::Col<T> coefs;
  arma::mat freqs;
  arma::Mat<bool> is_sin;

  inline
  Sinusoidal(const arma::Col<T> & coefs, const arma::mat & freqs) :
};
}
}
#endif //MATH_SINUSOIDAL_H
