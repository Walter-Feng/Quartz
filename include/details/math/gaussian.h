#ifndef MATH_GAUSSIAN_H
#define MATH_GAUSSIAN_H

#include "polynomial.h"

namespace quartz {
namespace math {
namespace gaussian {

// G(X) = Coef * exp(-1/2 X^T A X + B^T X)
template<typename T>
struct Term {
  Polynomial <T> polynomial;
  arma::mat binomial;
  arma::vec monomial;

  explicit
  inline
  Term(const arma::uword dim, const T coef = 0.0) :
      polynomial(Polynomial<T>(dim, coef)),
      binomial(arma::zeros<arma::mat>(dim, dim)),
      monomial(arma::zeros<arma::vec>(dim, dim)) {}

  inline
  Term(const Polynomial <T> polynomial,
       const arma::mat & binomial,
       const arma::vec & monomial) :
      polynomial(polynomial),
      binomial(binomial),
      monomial(monomial) {
    if (!binomial.is_square()) {
      throw Error("The binomial terms provided is not square");
    }
    if (binomial.n_rows != monomial.n_elem) {
      throw Error(
          "Different dimension between binomial term and monomial term");
    }
    if (binomial.n_rows != polynomial.dim()) {
      throw Error(
          "Different dimension between polynomial term and gaussian term"
      );
    }
  }

  inline
  Term(const arma::mat & binomial,
       const arma::vec & monomial,
       const T coef = 1.0) :
      polynomial(Polynomial<T>(binomial.n_rows, coef)),
      binomial(binomial),
      monomial(monomial) {
    if (!binomial.is_symmetric()) {
      throw Error("The binomial terms provided is not symmetric");
    }
    if (binomial.n_rows != monomial.n_elem) {
      throw Error(
          "Different dimension between binomial term and monomial term");
    }
  }

  explicit
  inline
  Term(const arma::mat & binomial, const T coef = 1.0) :
      polynomial(Polynomial<T>(dim, coef)),
      binomial(binomial),
      monomial(arma::zeros<arma::vec>(binomial.n_rows)) {}

  inline
  arma::uword dim() const {
    return monomial.n_elem;
  }

  template<typename U>
  std::common_type_t<T, U> at(const arma::Col<U> & position) const {
    if (this->dim() != position.n_elem) {
      throw Error(
          "Different dimension between the position and the gaussian term");
    }

    return this->polynomial.at(position) *
           std::exp(-0.5 * arma::dot(position, this->binomial * position) +
                    arma::dot(this->monomial, position));
  }

  inline
  Term<T> derivative(const arma::uword index) const {
    if (index >= this->dim) {
      throw Error("Derivative operator out of bound");
    }


    const Polynomial<T> contribution_from_gaussian =
        Polynomial<T>(-this->binomial.col(index),
                      arma::eye<lmat>(this->dim(), this->dim())) +
        this->monomial(index);

    return Term<T>(
        this->polynomial.derivative(index) + contribution_from_gaussian,
        this->binomial,
        this->monomial);
  }

  template<typename U>
  Term<std::common_type_t<T, U>>
  operator*(const Term<T> & B) const {
    if (this->dim() != B.dim()) {
      throw Error("Different dimension between multiplied gaussian terms");
    }
    return Term<T>(this->polynomial * B.polynomial, this->binomial + B.binomial,
                   this->monomial + B.monomial);
  }

  template<typename U>
  Term<std::common_type_t<T, U>>
  operator*(const polynomial::Term<U> & B) const {
    if (this->dim() != B.dim()) {
      throw Error("Different dimension between gaussian term and polynomial term");
    }
    return Term<T>(this->polynomial * B, this->binomial + B.binomial,
                   this->monomial + B.monomial);
  }

  template<typename U>
  Term<std::common_type_t<T,U>>
  operator*(const Polynomial<U> & B) const {
    if (this->dim() != B.dim()) {
      throw Error("Different dimension between gaussian term and polynomial term");
    }
    return Term<T>(this->polynomial * B, this->binomial + B.binomial,
                   this->monomial + B.monomial);
  }
  template<typename U>
  Term<std::common_type_t<T, U>>
  operator*(const U B) const {
    return Term<T>(this->polynomial * polynomial::Term<T>(this->dim(), B),
                   this->binomial + B.binomial,
                   this->monomial + B.monomial);
  }
};

}

template<typename T>
struct Gaussian {
  Polynomial <T> polynomial;
  arma::mat binomial;
  arma::vec monomial;
};
}
}
#endif //MATH_GAUSSIAN_H
