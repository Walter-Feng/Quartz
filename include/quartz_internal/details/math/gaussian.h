#ifndef MATH_GAUSSIAN_H
#define MATH_GAUSSIAN_H

#include "polynomial.h"
#include "exponential.h"

#include "quartz_internal/util/elementary_function_operator.h"

namespace math {

template<typename T>
struct Gaussian {
  T coef;
  arma::mat binomial;
  arma::Col<T> monomial;

  inline
  Gaussian(const arma::uword dim, const T coef = T{0.0}) :
      coef(coef),
      binomial(arma::zeros<arma::mat>(dim, dim)),
      monomial(arma::zeros<arma::vec>(dim, dim)) {}


  inline
  Gaussian(const arma::mat & binomial) :
      coef(T{1.0}),
      binomial(binomial),
      monomial(arma::zeros<arma::vec>(binomial.n_rows, binomial.n_rows)) {
    if (!binomial.is_symmetric()) {
      throw Error("The binomial term is not symmetric");
    }
  }

  inline
  Gaussian(const arma::mat & binomial, const arma::Col<T> & monomial,
           const T coef = T{1.0}) :
      coef(coef),
      binomial(binomial),
      monomial(monomial) {
    if (binomial.n_rows != monomial.n_elem) {
      throw Error("Different dimension between the binomial and monomial term");
    }
    if (!binomial.is_symmetric()) {
      throw Error("The binomial term is not symmetric");
    }
  }


  inline
  arma::uword dim() const {
    return this->binomial.n_rows;
  }

  template<typename U>
  std::common_type_t<T, U> at(const arma::Col<U> & position) const {
    if (position.n_elem != this->dim()) {
      throw Error("Different dimension between the gaussian term and position");
    }

    return this->coef * std::exp(
        -0.5 * arma::dot(position, this->binomial * position) +
        arma::dot(position, this->monomial));
  }

  inline
  T integral() const {

    return this->coef *
           std::sqrt(std::pow(2 * pi, this->dim()) / arma::det(this->binomial))
           *
           std::exp(0.5 * arma::dot(this->monomial, arma::inv(this->binomial) *
                                                    this->monomial));
  }

  template<typename U>
  std::common_type_t<T, U> integral(const Polynomial<U> & polynomial) const {

    if (polynomial.dim() != this->dim()) {
      throw Error(
          "Different dimension between the gaussian term and polynomial term");
    }


    const arma::mat inv_binomial = arma::inv(this->binomial);

    const auto functor = [&inv_binomial](
        const Polynomial<T> & poly) -> Polynomial<T> {
      Polynomial<T> result = Polynomial<T>(poly.dim());

      for (arma::uword i = 0; i < poly.dim(); i++) {
        for (arma::uword j = 0; j < poly.dim(); j++) {
          result = result +
                   poly.derivative(i).derivative(j) * 0.5 * inv_binomial(i, j);
        }
      }

      return result;
    };

    const auto post_functor = [this](const Polynomial<T> & poly) {

      return poly.at(this->monomial);
    };
    const std::common_type_t<T, U> polynomial_part = exp(functor, post_functor,
                                                         polynomial,
                                                         polynomial.grade() /
                                                         2);

    return this->coef *
           std::sqrt(std::pow(2 * pi, this->dim()) / arma::det(this->binomial))
           *
           std::exp(0.5 * arma::dot(this->monomial, arma::inv(this->binomial) *
                                                    this->monomial)) *
           polynomial_part;
  }


  inline
  arma::Col<T> mean() const {
    return arma::inv(this->binomial) * this->monomial;
  }

  inline
  arma::mat covariance() const {
    return arma::inv(this->binomial);
  }

  inline
  arma::vec variance() const {
    return this->covariance().diag();
  }

  inline
  Gaussian<double> wigner_transform() const {
    arma::vec eigenvalues;
    arma::mat eigenvectors;

    arma::eig_sym(eigenvalues, eigenvectors, this->binomial);

    const arma::mat transformed_binomial =
        arma::diagmat<arma::mat>(1. / eigenvalues);

    const arma::mat zero_matrix = arma::zeros<arma::mat>(
        arma::size(this->binomial));
    const arma::mat real_space_binomial_part = 2 * this->binomial;
    const arma::vec real_space_monomial_part = 2 * arma::real(this->monomial);
    const arma::mat momentum_space_binomial_part =
        2 * eigenvectors * transformed_binomial * eigenvectors.t();
    const arma::mat momentum_space_monomial_part =
        momentum_space_binomial_part * arma::imag(this->monomial);

    const arma::mat new_binomial =
        arma::join_cols(arma::join_rows(real_space_binomial_part, zero_matrix),
                        arma::join_rows(zero_matrix,
                                        momentum_space_binomial_part));

    const arma::mat new_monomial = arma::join_cols(real_space_monomial_part,
                                                   momentum_space_monomial_part);

    const double constant_part =
        1. / std::sqrt(arma::prod(eigenvalues)) *
        std::pow(1.0 / pi, this->dim() / 2.0) / std::exp(
            arma::dot(arma::imag(this->monomial),
                      momentum_space_binomial_part *
                      arma::imag(this->monomial)));

    return Gaussian<double>(new_binomial, new_monomial,
                            this->coef * constant_part);
  }

  template<typename U>
  Gaussian<std::common_type_t<T, U>> operator*(const U B) const {
    return Gaussian<std::common_type_t<T, U>>(this->binomial, this->monomial,
                                              this->coef * B);
  }

  template<typename U>
  Gaussian<std::common_type_t<T, U>>
  operator*(const exponential::Term<U> & B) const {
    return Gaussian<std::common_type_t<T, U>>(this->binomial,
                                              this->monomial + B.wavenumbers,
                                              this->coef * B.coef);
  }

  template<typename U>
  Gaussian<std::common_type_t<T, U>>
  operator*(const Gaussian<U> & B) const {
    return Gaussian<std::common_type_t<T, U>>(this->binomial + B.binomial,
                                              this->monomial + B.monomial,
                                              this->coef * B.coef);
  }

};

// G(X) = Coef * exp(-1/2 X^T A X + B^T X)
template<typename T>
struct GaussianWithPoly {
  Polynomial<T> polynomial;
  Gaussian<T> gaussian;

  explicit
  inline
  GaussianWithPoly(const arma::uword dim, const T coef = 0.0) :
      polynomial(Polynomial<T>(dim, coef)),
      gaussian(Gaussian<T>(dim, 1.0)) {}

  inline
  GaussianWithPoly(const Polynomial<T> polynomial,
                   const arma::mat & binomial,
                   const arma::Col<T> & monomial) :
      polynomial(polynomial),
      gaussian(Gaussian<T>(binomial, monomial, 1.0)) {
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
  GaussianWithPoly(const arma::mat & binomial,
                   const arma::Col<T> & monomial,
                   const T coef = 1.0) :
      polynomial(Polynomial<T>(binomial.n_rows, coef)),
      gaussian(binomial, monomial, 1.0) {
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
  GaussianWithPoly(const arma::mat & binomial, const T coef = 1.0) :
      polynomial(Polynomial<T>(binomial.n_cols, coef)),
      gaussian(Gaussian<T>(binomial, 1.0)) {}

  explicit
  inline
  GaussianWithPoly(const Gaussian<T> & gaussian) :
      polynomial(Polynomial<T>(gaussian.dim(), gaussian.coef)),
      gaussian(gaussian) {
    this->gaussian.coef = 1.0;
  }

  inline
  arma::uword dim() const {
    return this->gaussian.monomial.n_elem;
  }

  inline
  arma::uword grade() const {
    return this->polynomial.grade();
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
  GaussianWithPoly<T> derivative(const arma::uword index) const {
    if (index >= this->dim) {
      throw Error("Derivative operator out of bound");
    }


    const Polynomial<T> contribution_from_gaussian =
        Polynomial<T>(-this->binomial.col(index),
                      arma::eye<lmat>(this->dim(), this->dim())) +
        this->monomial(index);

    return GaussianWithPoly<T>(
        this->polynomial.derivative(index) + contribution_from_gaussian,
        this->binomial,
        this->monomial);
  }

  inline
  GaussianWithPoly<T> derivative(const arma::uvec & index) const {
    if (index.n_elem != this->dim) {
      throw Error("Derivative operator out of bound");
    }

    GaussianWithPoly<T> result = *this;

    for (arma::uword i = 0; i < index.n_elem; i++) {
      for (arma::uword j = 0; j < index(i); j++) {
        result = result.derivative(i);
      }
    }

    return result;

  }

  template<typename U>
  GaussianWithPoly<std::common_type_t<T, U>>
  operator*(const GaussianWithPoly<U> & B) const {
    if (this->dim() != B.dim()) {
      throw Error("Different dimension between multiplied gaussian terms");
    }
    return GaussianWithPoly<std::common_type_t<T, U>>(
        this->polynomial * B.polynomial,
        this->binomial + B.binomial,
        this->monomial + B.monomial);
  }

  template<typename U>
  GaussianWithPoly<std::common_type_t<T, U>>
  operator*(const polynomial::Term<U> & B) const {
    if (this->dim() != B.dim()) {
      throw Error(
          "Different dimension between gaussian term and polynomial term");
    }
    return GaussianWithPoly<T>(this->polynomial * B,
                               this->binomial,
                               this->monomial);
  }

  template<typename U>
  GaussianWithPoly<std::common_type_t<T, U>>
  operator*(const Polynomial<U> & B) const {
    if (this->dim() != B.dim()) {
      throw Error(
          "Different dimension between gaussian term and polynomial term");
    }
    return GaussianWithPoly<T>(this->polynomial * B,
                               this->binomial,
                               this->monomial);
  }

  template<typename U>
  GaussianWithPoly<std::common_type_t<T, U>>
  operator*(const U B) const {
    return GaussianWithPoly<std::common_type_t<T, U>>
        (this->polynomial * (this->dim(), B),
         this->binomial + B.binomial,
         this->monomial + B.monomial);
  }
};


//template<typename T>
//struct Gaussian {
//  std::vector<gaussian::Term<T>> terms;
//
//  explicit
//  inline
//  Gaussian(std::vector<gaussian::Term<T>> terms) :
//      terms(terms) {}
//
//  explicit
//  inline
//  Gaussian(const arma::uword dim, const T coef = 0.0) :
//      terms({gaussian::Term<T>(dim, coef)}) {}
//
//  inline
//  Gaussian(const Polynomial <T> polynomial,
//           const arma::mat & binomial,
//           const arma::vec & monomial) :
//      terms({gaussian::Term<T>(polynomial, binomial, monomial)}) {}
//
//  inline
//  Gaussian(const arma::mat & binomial,
//           const arma::vec & monomial,
//           const T coef = 1.0) :
//      terms({gaussian::Term<T>(binomial,monomial,coef)}) {}
//
//  explicit
//  inline
//  Gaussian(const arma::mat & binomial, const T coef = 1.0) :
//      terms({gaussian::Term<T>(binomial, coef)}) {}
//
//
//
//};

}

#endif //MATH_GAUSSIAN_H
