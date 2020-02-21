#ifndef MATH_GAUSSIAN_H
#define MATH_GAUSSIAN_H

#include "polynomial.h"
#include "exponential.h"

namespace math {
namespace gaussian {

// G(X) = Coef * exp(-1/2 X^T A X + B^T X)
template<typename T>
struct Term {
  Polynomial<T> polynomial;
  arma::mat binomial;
  arma::vec monomial;

  explicit
  inline
  Term(const arma::uword dim, const T coef = 0.0) :
      polynomial(Polynomial<T>(dim, coef)),
      binomial(arma::zeros<arma::mat>(dim, dim)),
      monomial(arma::zeros<arma::vec>(dim, dim)) {}

  inline
  Term(const Polynomial<T> polynomial,
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

  inline
  Term<T> derivative(const arma::uvec & index) const {
    if (index.n_elem != this->dim) {
      throw Error("Derivative operator out of bound");
    }

    Term<T> result = *this;

    for (arma::uword i = 0; i < index.n_elem; i++) {
      for (arma::uword j = 0; j < index(i); j++) {
        result = result.derivative(i);
      }
    }

    return result;

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
      throw Error(
          "Different dimension between gaussian term and polynomial term");
    }
    return Term<T>(this->polynomial * B, this->binomial + B.binomial,
                   this->monomial + B.monomial);
  }

  template<typename U>
  Term<std::common_type_t<T, U>>
  operator*(const Polynomial<U> & B) const {
    if (this->dim() != B.dim()) {
      throw Error(
          "Different dimension between gaussian term and polynomial term");
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

  Term<T> wigner_transform() const {
    arma::vec eigval;
    arma::mat eigvec;
    arma::eig_sym(eigval, eigvec, this->binomial);

    const arma::mat real_space_binomial = 2 * this->binomial;
    const arma::vec real_space_monomial = 2 * this->monomial;

    const double constant_part = 1.0 / pow(pi, this->dim()) * arma::det(eigvec);


  }
};

}

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

struct Gaussian {
  cx_double coef;
  arma::mat binomial;
  arma::vec monomial;
  exponential::Phase phase;

  inline
  Gaussian(const arma::uword dim, const cx_double coef = 0.0) :
      coef(coef),
      binomial(arma::zeros<arma::mat>(dim, dim)),
      monomial(arma::zeros<arma::vec>(dim, dim)),
      phase(exponential::Phase(dim)) {}


  inline
  Gaussian(const arma::mat & binomial, const cx_double coef = 1.0) :
      coef(1.0),
      binomial(binomial),
      monomial(arma::zeros<arma::vec>(binomial.n_rows, binomial.n_rows)),
      phase(exponential::Phase(binomial.n_rows)) {
    if (!binomial.is_symmetric()) {
      throw Error("The binomial term is not symmetric");
    }
  }

  inline
  Gaussian(const arma::mat & binomial, const arma::vec & monomial,
           const cx_double coef = 1.0) :
      coef(coef),
      binomial(binomial),
      monomial(monomial),
      phase(exponential::Phase(binomial.n_rows)) {
    if (binomial.n_rows != monomial.n_elem) {
      throw Error("Different dimension between the binomial and monomial term");
    }
    if (!binomial.is_symmetric()) {
      throw Error("The binomial term is not symmetric");
    }
  }

  inline
  Gaussian(const arma::mat & binomial, const arma::vec & monomial,
           const exponential::Phase & phase,
           const cx_double coef = 1.0) :
      coef(coef),
      binomial(binomial),
      monomial(monomial),
      phase(phase) {
    if (binomial.n_rows != monomial.n_elem) {
      throw Error("Different dimension between the binomial and monomial term");
    }

    if (binomial.n_rows != phase.wavenumbers.n_elem) {
      throw Error("Different dimension between the binomial and phase factor");
    }
    if (!binomial.is_symmetric()) {
      throw Error("The binomial term is not symmetric");
    }
  }

  inline
  arma::uword dim() const {
    return this->binomial.n_rows;
  }

  arma::cx_double at(const arma::vec & position) const {
    if (position.n_elem != this->dim()) {
      throw Error("Different dimension between the gaussian term and position");
    }

    return this->coef * std::exp(
        -0.5 * arma::cdot(position, this->binomial * position) +
        arma::cdot(position, this->monomial)) * this->phase.at(position);
  }

  inline
  cx_double integral() const {

    const arma::cx_vec new_monomial =
        this->monomial + cx_double{0.0, 1.0} * this->phase.wavenumbers;

    return this->coef *
           std::sqrt(std::pow(2 * pi, this->dim()) / arma::det(this->binomial))
           * std::exp(0.5 * arma::dot(new_monomial, arma::inv(this->binomial) *
                                                    new_monomial));
  }

  inline
  arma::cx_vec mean() const {
    const arma::cx_vec new_monomial =
        this->monomial + cx_double{0.0, 1.0} * this->phase.wavenumbers;

    return arma::inv(this->binomial) * new_monomial;
  }

  inline
  arma::mat covariance() const {
    return arma::inv(this->binomial);
  }

  inline
  Gaussian conj() const {
    return Gaussian(this->binomial, this->monomial, this->phase.conj(),
                    std::conj(this->coef));
  }

  inline
  Gaussian wigner_transform() const {
    arma::vec eigenvalues;
    arma::mat eigenvectors;

    arma::eig_sym(eigenvalues, eigenvectors, this->binomial);

    const arma::mat transformed_binomial =
        arma::diagmat<arma::mat>(1. / eigenvalues);

    const arma::mat zero_matrix = arma::zeros<arma::mat>(
        arma::size(this->binomial));
    const arma::mat real_space_binomial_part = 2 * this->binomial;
    const arma::vec real_space_monomial_part = 2 * this->monomial;
    const arma::mat momentum_space_binomial_part =
        2 * eigenvectors * transformed_binomial * eigenvectors.t();
    const arma::mat momentum_space_monomial_part =
        momentum_space_binomial_part * this->phase.wavenumbers;

    const arma::mat new_binomial =
        arma::join_cols(arma::join_rows(real_space_binomial_part, zero_matrix),
                        arma::join_rows(zero_matrix,
                                        momentum_space_binomial_part));

    const arma::mat new_monomial = arma::join_cols(real_space_monomial_part,
                                                   momentum_space_monomial_part);

    const double constant_part =
        1. / std::sqrt(arma::prod(eigenvalues)) *
        std::pow(1.0 / pi, this->dim() / 2.0) / std::exp(
            arma::dot(this->phase.wavenumbers,
                      momentum_space_binomial_part * this->phase.wavenumbers));

    return Gaussian(new_binomial, new_monomial, this->coef * constant_part);
  }

  Gaussian operator*(const cx_double B) const {
    return Gaussian(this->binomial, this->monomial, this->phase,
                    this->coef * B);
  }

  Gaussian operator*(const exponential::Term<double> & B) const {
    return Gaussian(this->binomial, this->monomial + B.wavenumbers, this->phase,
                    this->coef * B.coef);
  }

  Gaussian operator*(const exponential::Term<cx_double> & B) const {
    return Gaussian(this->binomial, this->monomial + arma::real(B.wavenumbers),
                    this->phase * exponential::Phase(arma::imag(B.wavenumbers)),
                    this->coef * B.coef);
  }

  Gaussian operator*(const Gaussian & B) const {
    return Gaussian(this->binomial + B.binomial, this->monomial + B.monomial,
                    this->phase * B.phase, this->coef * B.coef);
  }

};
}

#endif //MATH_GAUSSIAN_H
