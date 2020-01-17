#ifndef MATH_POLYNOMIAL_H
#define MATH_POLYNOMIAL_H

#include <armadillo>
#include <cassert>

#include "error.h"

namespace quartz {
namespace math {

// Term represents the each term a polynomial will have.
template<typename T>
struct Term {
  T coef;
  arma::uvec indices;

  template<typename U>
  U at(const arma::Col<U> & position) const {

    assert(position.n_elem == this->indices.n_elem);

    U result = u;

    for(arma::uword i=0;i<position.n_elem;i++) {
      result *= std::pow(position(i), this->indices(i));
    }

    return this->coef * result;
  }

  inline
  double at(const arma::vec & position) const {
    return this->at<double>(position);
  }

  inline
  arma::cx_double at(const arma::cx_vec & position) const {
    return this->at<arma::cx_double> (position);
  }

  template<typename U>
  arma::Col<U> at(const arma::Mat<U> & positions) const {

    assert(positions.n_rows == this->indices.n_elem);

    arma::Col<U> result = arma::zeros<arma::Col<U>>(positions.n_cols);

    for(arma::uword i=0;i<positions.n_cols;i++) {
      result(i) = this->at(positions.col(i));
    }

    return this->coef * result;
  }

  inline
  arma::vec at(const arma::mat & positions) const {
    return this->at<double>(positions);
  }

  inline
  arma::cx_vec at(const arma::cx_mat & positions) const {
    return this->at<arma::cx_double>(positions);
  }

  arma::uword dim() const {
    return this->indices.n_elem;
  }

  template<typename U>
  bool is_same_term(const Term<U> & term) {
    if(this->indices == term.indices) return true;

    return false;
  }

  inline
  bool is_same_term(const Term<double> & term) {
    return is_same_term<double>(term);
  }

  inline
  bool is_same_term(const Term<arma::cx_double> & term) {
    return is_same_term<arma::cx_double>(term);
  }

  bool operator == (const Term<T> & term) {
    return this->coef == term.coef && this->is_same_term(term);
  }
};


// The Polynomial struct is stored as a list of indices and the corresponding
// coefficients. The indices are stored column-wise.

template<typename T>
struct Polynomial {
public:
  inline
  Polynomial(const arma::Col<T> & coefs, const arma::umat & indices) :
      coefs(coefs),
      indices(indices)
  {
    if(coefs.n_elem != indices.n_cols) {
      throw Error("the number of coefficients and the indices are not consistent.");
    }
  }

  explicit
  inline
  Polynomial(const Term<T> term) :
      coefs(arma::Col<T>(term.coef)),
      indices(arma::umat(term.indices)) { }

  arma::umat indices;
  arma::Col<T> coefs;

  Term<T> term(arma::uword index) const {
    return Term<T>{this->coefs(index), this->indices.col(index)};
  }

  template<typename U>
  U at(const arma::Col<U> & position) const {
    assert(position.n_elem == this->indices.n_rows);

    U result = 0;

    for(arma::uword i=0;i<this->indices.n_cols;i++) {
      const Term<T> term = this->term(i);
      result += term.at(position);
    }

    return result;
  }

  inline
  double at(const arma::vec & position) const {
    return this->at<double>(position);
  }

  inline
  arma::cx_double at(const arma::cx_vec & position) const {
    return this->at<arma::cx_double> (position);
  }

  template<typename U>
  arma::Col<U> at(const arma::Mat<U> & positions) const {

    assert(positions.n_rows == this->indices.n_rows);

    arma::Col<U> result = arma::zeros<arma::Col<U>>(positions.n_cols);

    for(arma::uword i=0;i<positions.n_cols;i++) {
      result(i) = this->at(positions.col(i));
    }

    return this->coef * result;
  }

  inline
  arma::vec at(const arma::mat & positions) const {
    return this->at<double>(positions);
  }

  inline
  arma::cx_vec at(const arma::cx_mat & positions) const {
    return this->at<arma::cx_double>(positions);
  }

  inline
  Polynomial<T> derivative(const arma::uword index) const {
    for(arma::uword i=0;i<this->indices.n_rows;i++) {

    }
  }

  inline
  arma::uword dim() const {
    return this->indices.n_rows;
  }

  Polynomial<T> operator + (const Polynomial<T> & B) const {
    const arma::umat new_indices = arma::join_rows(this->indices, B->indices);
    const arma::Col<T> new_coefs = arma::join_cols(this->coefs, B->coefs);

    return {new_indices,new_coefs};
  }

  Polynomial<T> operator + (const double B) const {

    const arma::uvec dummy_indices = arma::zeros<arma::uvec>(this->indices.n_rows);
    const arma::umat new_indices = arma::join_rows(this->indices, dummy_indices);
    const arma::Col<T> new_coefs = arma::join_cols(this->coefs, B);
  }

};


}
}

#endif //MATH_POLYNOMIAL_H
