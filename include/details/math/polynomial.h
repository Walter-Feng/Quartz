#ifndef MATH_POLYNOMIAL_H
#define MATH_POLYNOMIAL_H

#include <armadillo>

#include "alias.h"
#include "error.h"

#include "util/type_converter.h"

namespace quartz {
namespace math {
namespace polynomial {

// Term represents the each term a polynomial will have.
template<typename T>
struct Term {
  T coef;
  lvec indices;

  template<typename U>
  std::common_type_t<T, U> at(const arma::Col<U> & position) const {

    if (position.n_elem != this->indices.n_elem) {
      throw Error(
          "Different dimension between the position and polynomial term");
    }

    auto result = std::common_type_t<T, U>(1.0);

    for (arma::uword i = 0; i < position.n_elem; i++) {
      result *= std::pow(position(i), this->indices(i));
    }

    return this->coef * result;
  }

  arma::uword dim() const {
    return this->indices.n_elem;
  }

  template<typename U>
  bool is_same_term(const Term<U> & term) const {
    if (this->indices == term.indices) return true;

    return false;
  }

  inline
  Term<T> derivative(const arma::uword index) const {
    if (this->indices(index) == 0) {
      return {0., arma::zeros<arma::Col<T>>(this->dim())};
    } else {
      lvec new_indices = this->indices;
      new_indices(index) -= 1;
      return {this->coef * (new_indices(index) + 1), new_indices};
    }
  }

  inline
  Term<T> derivative(const arma::uvec & index) const {
    if (index >= this->indices.n_elem) {
      throw Error("Derivative operator out of bound");
    }

    Term<T> result = *this;
    for (arma::uword i = 0; i < index.n_elem; i++) {
      for (arma::uword j = 0; j < index(i); j++) {
        result = result.derivative(j);
      }
    }

    return result;
  }

  bool operator==(const Term<T> & term) const {
    return this->coef == term.coef && this->is_same_term(term);
  }
};

} // namespace polynomial

// The Polynomial struct is stored as a list of indices and the corresponding
// coefficients. The indices are stored column-wise.

template<typename T>
struct Polynomial {
public:
  inline
  Polynomial(const arma::Col<T> & coefs, const lmat & indices) :
      coefs(coefs),
      indices(indices) {
    if (coefs.n_elem != indices.n_cols) {
      throw Error(
          "the number of coefficients and the indices are not consistent");
    }
  }

  explicit
  inline
  Polynomial(const polynomial::Term<T> term) :
      coefs(arma::Col<T>(term.coef)),
      indices(lmat(term.indices)) {}

  arma::Col<T> coefs;
  lmat indices;

  inline
  polynomial::Term<T> term(arma::uword index) const {
    return polynomial::Term<T>{this->coefs(index), this->indices.col(index)};
  }

  template<typename U>
  std::common_type_t<T, U> at(const arma::Col<U> & position) const {

    if (position.n_elem != this->indices.n_rows) {
      throw Error(
          "Different dimension between the position and polynomial term");
    };

    auto result = std::common_type_t<T, U>(0.0);

    for (arma::uword i = 0; i < this->indices.n_cols; i++) {
      const polynomial::Term<T> term = this->term(i);
      result += term.at(position);
    }

    return result;
  }

  inline
  Polynomial<T> derivative(const arma::uword index) const {
    Polynomial<T> result = Polynomial<T>(this->term(0).derivative(index));

    for (arma::uword i = 0; i < this->coefs.n_elem; i++) {
      result = result + this->term(i).derivative(index);
    }

    return result;
  }

  inline
  Polynomial<T> derivative(const arma::uvec & index) const {
    if (index >= this->dim()) {
      throw Error("Derivative operator out of bound");
    }

    Polynomial<T> result = *this;
    for (arma::uword i = 0; i < index.n_elem; i++) {
      for (arma::uword j = 0; j < index(i); j++) {
        result = result.derivative(j);
      }
    }

    return result;
  }

  template<typename U>
  Polynomial<std::common_type_t<T, U>>
  operator+(const Polynomial<U> & B) const {
    const lmat new_indices = arma::join_rows(this->indices, B.indices);
    const auto converted_this_coefs =
        arma::conv_to<arma::Col<std::common_type_t<T, U>>>::from(this->coefs);
    const auto converted_B_coefs =
        arma::conv_to<arma::Col<std::common_type_t<T, U>>>::from(B.coefs);
    const arma::Col<std::common_type_t<T, U>>
        new_coefs =
            arma::join_cols(converted_this_coefs,converted_B_coefs);

    return {new_coefs, new_indices};
  }

  template<typename U>
  Polynomial<std::common_type_t<T, U>> operator+(const U B) const {

    const lvec dummy_indices = arma::zeros<lvec>(
        this->indices.n_rows);
    const lmat new_indices = arma::join_rows(this->indices,
                                             dummy_indices);
    const arma::Col<std::common_type_t<T, U>> new_coefs = arma::join_cols(
        this->coefs, B);

    return {new_coefs, new_indices};
  }

  template<typename U>
  Polynomial<std::common_type_t<T, U>>
  operator+(const polynomial::Term<U> & B) const {
    const lmat new_indices = arma::join_rows(this->indices, B.indices);
    const auto converted_this_coefs =
        arma::conv_to<arma::Col<std::common_type_t<T, U>>>::from(this->coefs);
    const auto converted_B_coef = arma::Col<std::common_type_t<T,U>>{B.coef};
    const arma::Col<std::common_type_t<T, U>>
        new_coefs = arma::join_cols(converted_this_coefs, converted_B_coef);

    return {new_coefs, new_indices};
  }

  template<typename U>
  Polynomial<std::common_type_t<T, U>>
  operator*(const polynomial::Term<U> & B) const {
    lmat new_indices = this->indices;
    new_indices.each_col() += B.indices;
    const arma::Col<std::common_type_t<T, U>>
        new_coefs = this->coefs * B.coef;

    return {new_coefs, new_indices};
  }

  template<typename U>
  Polynomial<std::common_type_t<T,U>> operator*(const Polynomial<U> & B) const {
    Polynomial<std::common_type_t<T,U>> result_0 = (*this) * B.term(0);

    for (arma::uword i = 1; i < B.coefs.n_elem; i++) {
      result_0 = result_0 + (*this) * B.term(i);
    }

    return result_0;
  }

  template<typename U>
  Polynomial<std::common_type_t<T, U>> operator*(const U B) const {
    return {this->coefs * B, this->indices};
  }

  template<typename U>
  Polynomial<std::common_type_t<T, U>>
  operator-(const Polynomial<U> & B) const {
    return *this + B * (-1.0);
  }

  template<typename U>
  Polynomial<std::common_type_t<T, U>> operator-(const U B) const {

    const lvec dummy_indices = arma::zeros<lvec>(
        this->indices.n_rows);
    const lmat new_indices = arma::join_rows(this->indices,
                                             dummy_indices);
    const arma::Col<std::common_type_t<T, U>>
        new_coefs = arma::join_cols(this->coefs, -B);

    return {new_coefs, new_indices};
  }

  template<typename U>
  Polynomial<std::common_type_t<T, U>> operator/(const U B) const {
    return *(this) * (1.0 / B);
  }

  template<typename U>
  Polynomial<T> operator/(const polynomial::Term<T> & B) const {
    lmat new_indices = this->indices;
    new_indices.each_col() -= B.indices;
    const arma::Col<std::common_type_t<T,U>> new_coefs = this->coefs / B.coef;

    return {new_coefs, new_indices};
  }
};

}
}

#endif //MATH_POLYNOMIAL_H
