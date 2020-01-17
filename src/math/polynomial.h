#ifndef MATH_POLYNOMIAL_H
#define MATH_POLYNOMIAL_H

#include <armadillo>
#include <cassert>

namespace quartz {
namespace math {
namespace polynomial {

template<typename T>
struct Term {
  arma::uvec indices;
  arma::Col<T> coef;

  template<typename U>
  U at(const arma::Col<U> & position) {

    assert(position.n_elem == this->indices.n_elem);

    U result = 0;

    for(arma::uword i=0;i<position.n_elem;i++) {
      result += std::pow(position(i), this->indices(i));
    }

    return this->coef * result;
  }

  double at(const arma::vec & position){
    return this->at<double>(position);
  }

  arma::cx_double at(const arma::cx_vec & position) {
    return this->at<arma::cx_double> (position);
  }

};

template<typename T>
struct Polynomial {
  arma::umat indices;
  arma::Col<T> coefs;

  Term<T> term(arma::uword index) {
    return Term<T>{indices.col(index)};
  }

  template<typename U>
  U at(arma::Col<U> position) {
    assert(position.n_elem == this->indices.n_rows);

    U result = 0;

    for(arma::uword i=0;i<this->indices.n_rows;i++) {
      const Term<T> & term = this->term(i);
      result += term.at(position);
    }

    return result;
  }

  double at(const arma::vec & position){
    return this->at<double>(position);
  }

  arma::cx_double at(const arma::cx_vec & position) {
    return this->at<arma::cx_double> (position);
  }

  Polynomial derivative(const arma::uword index) {
    for(arma::uword i=0;i<this->indices.n_rows;i++) {

    }
  }

  Polynomial operator + (const Polynomial<T> & B) {
    const arma::umat new_indices = arma::join_rows(this->indices, B->indices);
    const arma::Col<T> new_coefs = arma::join_cols(this->coefs, B->coefs);
  }

  template<typename U>
};


}
}
}

#endif //MATH_POLYNOMIAL_H
