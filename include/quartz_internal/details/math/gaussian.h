#ifndef MATH_GAUSSIAN_H
#define MATH_GAUSSIAN_H

#include "polynomial.h"
#include "exponential.h"

#include "quartz_internal/util/elementary_function_operator.h"

namespace math {

template<typename T>
struct Gaussian {
  T coef;
  arma::Mat<T> covariance;
  arma::Col<T> center;

  inline
  Gaussian() :
      coef(0.0),
      covariance(arma::eye<arma::Mat<T>>(1, 1)),
      center(arma::Col<T>{1.0}) {}

  inline
  Gaussian(const Gaussian<T> & gaussian) :
      coef(gaussian.coef),
      covariance(gaussian.covariance),
      center(gaussian.center) {}

  inline
  Gaussian(const arma::uword dim, const T coef = T{0.0}) :
      coef(coef),
      covariance(arma::eye<arma::Mat<T>>(dim, dim)),
      center(arma::zeros<arma::vec>(dim)) {}


  inline
  Gaussian(const arma::Mat<T> & covariance, const T coef = T{1.0}) :
      coef(coef),
      covariance(covariance),
      center(arma::zeros<arma::vec>(covariance.n_rows)) {
    if (!covariance.is_symmetric()) {
      throw Error("The covariance term is not symmetric");
    }
  }

  inline
  Gaussian(const arma::Mat<T> & covariance, const arma::Col<T> & mean,
           const T coef = T{1.0}) :
      coef(coef),
      covariance(covariance),
      center(mean) {
    if (covariance.n_rows != mean.n_elem) {
      throw Error("Different dimension between the covariance and center term");
    }
    if (!covariance.is_symmetric()) {
      throw Error("The covariance term is not symmetric");
    }
  }

  inline
  Gaussian<cx_double> with_phase_factor(const arma::vec & phase_factor) const {
    if (this->dim() != phase_factor.n_elem) {
      throw Error(
          "Different dimension between the gaussian function and phase factor");
    }

    const arma::cx_vec new_mean =
        this->center + cx_double{0.0, 1.0} * this->covariance * phase_factor;
    return Gaussian<cx_double>(
        arma::conv_to<arma::cx_mat>::from(this->covariance),
        new_mean, this->coef);
  }

  inline
  arma::mat cov() const {
    return this->covariance;
  }

  inline
  arma::Col<T> mean() const {
    return this->center;
  }

  inline
  arma::uword dim() const {
    return this->covariance.n_rows;
  }

  inline
  Gaussian<double> abs() const {

    const arma::cx_mat cov_inv = arma::inv(this->covariance);
    const arma::mat real_cov = arma::inv(arma::real(cov_inv));
    const arma::vec new_B = arma::real(cov_inv * this->center);
    const arma::vec new_center = real_cov * new_B;
    const double new_coef = std::abs(this->coef) *
        std::exp(0.5 * arma::dot(new_center, new_B))
        / std::abs(
            std::exp(0.5 *
            arma::dot(this->center, arma::inv(this->covariance)* this->center)));
    return Gaussian<double>(real_cov, new_center, new_coef);
  }

  template<typename U>
  std::common_type_t<T, U> at(const arma::Col<U> & position) const {
    if (position.n_elem != this->dim()) {
      throw Error("Different dimension between the gaussian term and position");
    }

    return this->coef * std::exp(
        -0.5 * arma::dot(position - this->center,
                         arma::inv(this->covariance) *
                         (position - this->center)));
  }

  inline
  T integral() const {

    return this->coef *
           std::sqrt(
               std::pow(2 * pi, this->dim()) * arma::det(this->covariance));
  }

  template<typename U>
  std::common_type_t<T, U> expectation(const Polynomial<U> & polynomial) const {

    if (polynomial.dim() != this->dim()) {
      throw Error(
          "Different dimension between the gaussian term and polynomial term");
    }

    const Polynomial<std::common_type_t<T, U>>
    converted_polynomial(
        arma::conv_to<arma::Col<std::common_type_t<T,U>>>::from(polynomial.coefs),
        polynomial.exponents
        );

    const auto functor = [this](
        const Polynomial<std::common_type_t<T, U>> & poly) -> auto {
      auto result = Polynomial<std::common_type_t<T, U>>(poly.dim());

      for (arma::uword i = 0; i < poly.dim(); i++) {
        for (arma::uword j = 0; j < poly.dim(); j++) {
          result = result +
                   poly.derivative(i).derivative(j) * 0.5 *
                   this->covariance(i, j);
        }
      }

      return result;
    };

    const auto post_functor =
        [this](const Polynomial<std::common_type_t<T, U>> & poly) -> auto {
          return poly.at(this->center);
        };
    const std::common_type_t<T, U> polynomial_part = exp(functor, post_functor,
                                                         converted_polynomial,
                                                         polynomial.grade() / 2 + 1);

    return polynomial_part;
  }

  template<typename U>
  std::common_type_t<T, U> integral(const Polynomial<U> & polynomial) const {
    return this->expectation(polynomial) * this->integral();
  }

  inline
  arma::vec variance() const {
    return this->covariance().diag();
  }

  inline
  Gaussian<double> wigner_transform() const {

    if (!arma::approx_equal(arma::imag(this->covariance),
                            arma::zeros<arma::mat>(
                                arma::size(this->covariance)),
                            "abs_diff", 1e-16)) {
      throw Error(
          "Currently wigner transformation can only be applied to real covariance");
    }
    arma::vec eigenvalues;
    arma::mat eigenvectors;

    arma::eig_sym(eigenvalues, eigenvectors, arma::real(this->covariance));

    const arma::mat transformed_covariance =
        arma::diagmat<arma::mat>(1. / eigenvalues);

    const arma::mat zero_matrix = arma::zeros(arma::size(this->covariance));
    const arma::mat real_space_covariance_part =
        0.5 * arma::real(this->covariance);
    const arma::vec real_space_mean_part = arma::real(this->center);
    const arma::mat momentum_space_covariance_part =
        0.5 * eigenvectors * transformed_covariance * eigenvectors.t();
    const arma::vec momentum_space_mean_part =
        arma::imag(arma::inv(this->covariance) * this->center);
    const arma::mat new_covariance =
        arma::join_cols(
            arma::join_rows(real_space_covariance_part, zero_matrix),
            arma::join_rows(zero_matrix,
                            momentum_space_covariance_part));
    const arma::vec new_mean = arma::join_cols(real_space_mean_part,
                                               momentum_space_mean_part);
    const double constant_part =
        std::sqrt(arma::prod(eigenvalues)) *
        std::pow(1.0 / pi, this->dim() / 2.0);

    assert(std::imag(this->coef) == 0);

    return Gaussian<double>(new_covariance, new_mean,
                            std::real(this->coef) * constant_part);
  }

  template<typename U>
  Gaussian<std::common_type_t<T, U>> operator*(const U B) const {
    return Gaussian<std::common_type_t<T, U>>(this->covariance, this->center,
                                              this->coef * B);
  }

  template<typename U>
  Gaussian<std::common_type_t<T, U>>
  operator*(const exponential::Term<U> & B) const {
    return Gaussian<std::common_type_t<T, U>>(this->covariance,
                                              this->center + B.wavenumbers,
                                              this->coef * B.coef);
  }

  template<typename U>
  Gaussian<std::common_type_t<T, U>>
  operator*(const Gaussian<U> & B) const {

    const arma::mat this_A = arma::inv(this->covariance);
    const arma::mat B_A = arma::inv(B.covariance);

    const arma::mat A_sum = this_A + B_A;
    const arma::mat new_covariance = arma::inv(A_sum);

    const arma::Col<std::common_type_t<T, U>> new_mean =
        new_covariance * (this_A * this->center + B_A * B.center);

    const std::common_type_t<T, U> new_coef =
        std::exp(-0.5 * (arma::dot(this->center, this_A * this->center)
                         + arma::dot(B.center, B_A * B.center)
                         - arma::dot(new_mean, A_sum * new_mean))) *
        this->coef * B.coef;
    return
        Gaussian<std::common_type_t<T, U>>(new_covariance,
                                           new_mean,
                                           new_coef);
  }

};

template<typename T>
struct GaussianWithPoly {
  Polynomial<T> polynomial;
  Gaussian<T> gaussian;

  inline
  GaussianWithPoly(const GaussianWithPoly<T> & function) :
      polynomial(function.polynomial),
      gaussian(function.gaussian) {}

  inline
  GaussianWithPoly() :
      polynomial(),
      gaussian() {}

  inline
  GaussianWithPoly(const arma::uword dim, const T coef = 0.0) :
      polynomial(Polynomial<T>(dim, coef)),
      gaussian(Gaussian<T>(dim, 1.0)) {}

  inline
  GaussianWithPoly(const Polynomial<T> & polynomial,
                   const arma::Mat<T> & covariance,
                   const arma::Col<T> & mean) :
      polynomial(polynomial),
      gaussian(Gaussian<T>(covariance, mean, 1.0)) {
    if (!covariance.is_square()) {
      throw Error("The covariance terms provided is not square");
    }
    if (covariance.n_rows != mean.n_elem) {
      throw Error(
          "Different dimension between covariance term and center term");
    }
    if (covariance.n_rows != polynomial.dim()) {
      throw Error(
          "Different dimension between polynomial term and gaussian term"
      );
    }
  }

  inline
  GaussianWithPoly(const Polynomial<T> & polynomial,
                   const Gaussian<T> & gaussian) :
      polynomial(polynomial * gaussian.coef),
      gaussian(gaussian) {
    if (polynomial.dim() != gaussian.dim()) {
      throw Error(
          "Different dimension between polynomial term and gaussian term");
    }

    this->gaussian.coef = 1.0;
  }

  inline
  GaussianWithPoly(const arma::Mat<T> & covariance,
                   const arma::Col<T> & mean,
                   const T coef = 1.0) :
      polynomial(Polynomial<T>(covariance.n_rows, coef)),
      gaussian(covariance, mean, 1.0) {
    if (!covariance.is_symmetric()) {
      throw Error("The covariance terms provided is not symmetric");
    }
    if (covariance.n_rows != mean.n_elem) {
      throw Error(
          "Different dimension between covariance term and center term");
    }
  }


  inline
  GaussianWithPoly(const arma::Mat<T> & covariance, const T coef = 1.0) :
      polynomial(Polynomial<T>(covariance.n_cols, coef)),
      gaussian(Gaussian<T>(covariance, 1.0)) {}


  inline
  GaussianWithPoly(const Gaussian<T> & gaussian) :
      polynomial(Polynomial<T>(gaussian.dim(), gaussian.coef)),
      gaussian(gaussian) {
    this->gaussian.coef = 1.0;
  }

  inline
  arma::uword dim() const {
    return this->gaussian.center.n_elem;
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

    return this->polynomial.at(position) * this->gaussian.at(position);
  }

  inline
  GaussianWithPoly<T> derivative(const arma::uword index) const {
    if (index >= this->dim()) {
      throw Error("Derivative operator out of bound");
    }

    const arma::mat inv_covariance = arma::inv(this->gaussian.covariance);
    const arma::Col<T> B = inv_covariance * this->gaussian.center;

    const Polynomial<T> contribution_from_gaussian =
        (Polynomial<T>(-inv_covariance.col(index),
                       arma::eye<lmat>(this->dim(), this->dim())) + B(index))
        * this->polynomial;

    return GaussianWithPoly<T>(
        this->polynomial.derivative(index) + contribution_from_gaussian,
        this->gaussian);
  }

  inline
  GaussianWithPoly<T> derivative(const arma::uvec & index) const {
    if (index.n_elem != this->dim()) {
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

  inline
  T integral() const {
    return this->gaussian.integral(this->polynomial);
  }

  template<typename U>
  GaussianWithPoly<std::common_type_t<T, U>>
  operator+(const GaussianWithPoly<U> & B) const {
    if (this->dim() != B.dim()) {
      throw Error("Different dimension between multiplied gaussian terms");
    }
    if (!arma::approx_equal(this->gaussian.center, B.gaussian.center,
                            "abs_diff",
                            1e-16) ||
        !arma::approx_equal(this->gaussian.covariance, B.gaussian.covariance,
                            "abs_diff", 1e-16)) {
      throw Error(
          "GaussianWithPoly does not support adding with different gaussian terms");
    }
    return GaussianWithPoly<std::common_type_t<T, U>>(
        this->polynomial + B.polynomial, this->gaussian);
  }

  template<typename U>
  GaussianWithPoly<std::common_type_t<T, U>>
  operator-(const GaussianWithPoly<U> & B) const {
    return GaussianWithPoly<std::common_type_t<T, U>>(
        this->polynomial - B.polynomial, this->gaussian);
  }

  template<typename U>
  GaussianWithPoly<std::common_type_t<T, U>>
  operator*(const GaussianWithPoly<U> & B) const {
    if (this->dim() != B.dim()) {
      throw Error("Different dimension between multiplied gaussian terms");
    }
    return GaussianWithPoly<std::common_type_t<T, U>>(
        this->polynomial * B.polynomial, this->gaussian * B.gaussian);
  }

  template<typename U>
  GaussianWithPoly<std::common_type_t<T, U>>
  operator*(const polynomial::Term<U> & B) const {
    if (this->dim() != B.dim()) {
      throw Error(
          "Different dimension between gaussian term and polynomial term");
    }
    return GaussianWithPoly<T>(this->polynomial * B, this->gaussian);
  }

  template<typename U>
  GaussianWithPoly<std::common_type_t<T, U>>
  operator*(const Polynomial<U> & B) const {
    if (this->dim() != B.dim()) {
      throw Error(
          "Different dimension between gaussian term and polynomial term");
    }
    return GaussianWithPoly<T>(this->polynomial * B, this->gaussian);
  }

  template<typename U>
  GaussianWithPoly<std::common_type_t<T, U>>
  operator*(const U B) const {
    return GaussianWithPoly<std::common_type_t<T, U>>
        (this->polynomial * B, this->gaussian);
  }
};


//template<typename T>
//struct Gaussian {
//  std::vector<gaussian::Term<T>> terms;
//
//
//  inline
//  Gaussian(std::vector<gaussian::Term<T>> terms) :
//      terms(terms) {}
//
//
//  inline
//  Gaussian(const arma::uword dim, const T coef = 0.0) :
//      terms({gaussian::Term<T>(dim, coef)}) {}
//
//  inline
//  Gaussian(const Polynomial <T> polynomial,
//           const arma::mat & covariance,
//           const arma::vec & center) :
//      terms({gaussian::Term<T>(polynomial, covariance, center)}) {}
//
//  inline
//  Gaussian(const arma::mat & covariance,
//           const arma::vec & center,
//           const T coef = 1.0) :
//      terms({gaussian::Term<T>(covariance,center,coef)}) {}
//
//
//  inline
//  Gaussian(const arma::mat & covariance, const T coef = 1.0) :
//      terms({gaussian::Term<T>(covariance, coef)}) {}
//
//
//
//};

}

#endif //MATH_GAUSSIAN_H
