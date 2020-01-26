#ifndef MATH_SINUSOIDAL_H
#define MATH_SINUSOIDAL_H

#include "details/math/constants.h"

namespace quartz {
namespace math {

namespace sinusoidal {
template<typename T>
struct Term {
  T coef;
  arma::vec freqs;
  arma::vec is_sin;
  arma::vec translation;

  inline
  Term(const double coef,
       const arma::mat & freqs,
       const arma::Col<bool> & is_sin) :
       coef(coef),
       freqs(freqs),
       is_sin(0.5 * pi * (1 - arma::conv_to<arma::vec>::from(is_sin))),
       translation(arma::zeros<arma::vec>(arma::size(freqs))) {
    if (arma::size(freqs) != arma::size(is_sin)) {
      throw Error(
          "Different size between the frequencies and the is_sin flags");
    }
  }
};
}

template<typename T>
struct Sinusoidal {
  arma::Col <T> coefs;
  arma::mat freqs;
  arma::mat is_sin;
  arma::mat translation;

  inline
  Sinusoidal(const arma::Col <T> & coefs,
             const arma::mat & freqs,
             const arma::Mat<bool> & is_sin) :
      coefs(coefs),
      freqs(freqs),
      is_sin(0.5 * pi * (1 - arma::conv_to<arma::mat>::from(is_sin))),
      translation(arma::zeros<arma::mat>(arma::size(freqs))){
    if (coefs.n_elem != freqs.n_rows) {
      throw Error(
          "the number between the coefficients "
          "and the frequencies is not consistent");
    }

    if(arma::size(freqs) != arma::size(is_sin)) {
      throw Error("Different size between the frequencies and the is_sin flags");
    }
  }

  inline
  Sinusoidal(const arma::Col<T> & coefs,
             const arma::mat & freqs,
             const arma::mat & translation) :
     coefs(coefs),
     freqs(freqs),
     is_sin(arma::zeros<arma::mat>(arma::size(freqs))),
     translation(translation) {
       if (coefs.n_elem != freqs.n_cols) {
         throw Error(
             "the number between the coefficients "
             "and the frequencies is not consistent");
       }

       if(arma::size(freqs) != arma::size(translation)) {
         throw Error("Different size between the frequencies and the translations");
       }
  }

  explicit
  inline
  Sinusoidal(const arma::uword dim,
             const T coef = 0.0) :
      coefs(arma::Col<T>{coef}),
      freqs(arma::zeros<arma::mat>(dim,1)),
      is_sin(arma::zeros<arma::mat>(arma::size(freqs))),
      translation(translation) {
    if (coefs.n_elem != freqs.n_cols) {
      throw Error(
          "the number between the coefficients "
          "and the frequencies is not consistent");
    }

    if(arma::size(freqs) != arma::size(translation)) {
      throw Error("Different size between the frequencies and the translations");
    }
  }

  inline
  arma::uword dim() const {
    return this->freqs.n_rows;
  }

  inline
  T at(const arma::vec & position) const {
    arma::mat center = this->translation;
    center.each_col() += position;
    return arma::sum(
        this->coefs % arma::prod(arma::sin(center % this->freqs + is_sin)));
  }

  inline
  Sinusoidal<T> derivative(const arma::uvec & index) const {
    if(index.n_elem != this->dim()) {
      throw Error("Derivative operator out of bound");
    }

    arma::mat freq_change = this->freqs;
    freq_change.each_col() %= arma::conv_to<arma::vec>::from(index);
    const arma::Col<T> new_coefs = this->coefs % arma::prod(freq_change).t();
    arma::mat new_is_sin = this->is_sin;
    new_is_sin.each_col() += - 0.5 * pi * arma::conv_to<arma::vec>::from(index);

    return{new_coefs, this->freqs, new_is_sin, this->translation};
  }

  inline
  Sinusoidal<T> derivative(const arma::uword & index) const {
    if(index >= this->dim()) {
      throw Error("Derivative operator out of bound");
    }

    const arma::Col<T> new_coefs = this->coefs % this->freqs.col(index);
    arma::mat new_is_sin = this->is_sin;
    new_is_sin.col(index) += - 0.5 * pi;

    return{new_coefs, this->freqs, new_is_sin, this->translation};
  }

  template<typename U>
  Sinusoidal<std::common_type_t<T, U>>
  operator+(const Sinusoidal<U> & B) const {
    const lmat new_freqs = arma::join_rows(this->freqs, B.freqs);
    const auto converted_this_coefs =
        arma::conv_to<arma::Col<std::common_type_t<T, U>>>::from(this->coefs);
    const auto converted_B_coefs =
        arma::conv_to<arma::Col<std::common_type_t<T, U>>>::from(B.coefs);
    const arma::Col<std::common_type_t<T, U>>
        new_coefs =
        arma::join_cols(converted_this_coefs, converted_B_coefs);
    const auto new_translation = arma::join_rows(this->translation, B.translation);
    const auto new_is_sin = arma::join_rows(this->is_sin, B.is_sin);


    return {new_coefs, new_freqs, new_is_sin, new_translation};
  }

  template<typename U>
  Sinusoidal<std::common_type_t<T, U>>
  operator+(const sinusoidal::Term<T> & B) const {
    const lmat new_freqs = arma::join_rows(this->freqs, B.freqs);
    const auto converted_this_coefs =
        arma::conv_to<arma::Col<std::common_type_t<T, U>>>::from(this->coefs);
    const auto converted_B_coefs =
        arma::Col<std::common_type_t<T, U>>(B.coef);
    const arma::Col<std::common_type_t<T, U>>
        new_coefs =
        arma::join_cols(converted_this_coefs, converted_B_coefs);
    const auto new_translation = arma::join_rows(this->translation, B.translation);
    const auto new_is_sin = arma::join_rows(this->is_sin, B.is_sin);


    return {new_coefs, new_freqs, new_is_sin, new_translation};
  }
};
}
}
#endif //MATH_SINUSOIDAL_H
