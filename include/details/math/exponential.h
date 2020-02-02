#ifndef MATH_EXPONENTIAL_H
#define MATH_EXPONENTIAL_H

namespace quartz {
namespace math {
namespace exponential {

struct Phase {
  arma::vec wavenumbers;

  explicit
  inline Phase(const arma::uword dim) :
      wavenumbers(arma::zeros<arma::vec>(dim)) {}

  explicit
  inline Phase(const arma::vec & wavenumbers) :
      wavenumbers(wavenumbers) {}

  cx_double at(const arma::vec & position) const {
    if (position.n_elem != this->wavenumbers.n_elem) {
      throw Error(
          "Different dimension between the position and exponential term");
    }

    return std::exp(
        arma::sum((this->wavenumbers * cx_double{0.0, 1.0}) % position));
  }

  Phase operator*(const Phase & phase) const {
    const arma::vec new_wavenumbers = this->wavenumbers + phase.wavenumbers;
    return Phase(new_wavenumbers);
  }
};

template<typename T>
struct Term {
  T coef;
  arma::Col<T> wavenumbers;

  template<typename U>
  std::common_type_t<T, U> at(const arma::Col<U> & position) const {
    if (position.n_elem != this->wavenumbers.n_elem) {
      throw Error(
          "Different dimension between the position and exponential term");
    }

    return coef * std::exp(arma::sum(this->wavenumbers % position));
  }

  explicit
  inline
  Term(const arma::uword dim, const T coef = T(0.0)) :
      coef(coef),
      wavenumbers(arma::zeros<arma::Col<T>>(dim)) {}

  inline
  Term(const T coef, const arma::Col<T> & wavenumbers) :
      coef(coef),
      wavenumbers(wavenumbers) {}

  inline
  Term<T> derivative(const arma::uword index) const {
    return {this->wavenumbers(index) * this->coef, this->wavenumbers};
  }

  inline
  Term<T> derivative(const arma::uvec & index) const {
    return {arma::prod(this->wavenumbers % index) * index};
  }

  inline
  Phase phase() const {
    return Phase(arma::imag(this->wavenumbers));
  }

  template<typename U>
  Term<std::common_type_t<T, U>> operator*(const Term<U> & term) const {
    return {this->coef * term.coef, this->wavenumbers + term.wavenumbers};
  }

  template<typename U>
  Term<std::common_type_t<T, U>> operator/(const Term<U> & term) const {
    return {this->coef / term.coef, this->wavenumbers - term.wavenumbers};
  }

};
}

template<typename T>
struct Exponential {
  arma::Col<T> coefs;
  arma::Mat<T> wavenumbers;

  explicit
  inline Exponential(const arma::uword dim, const T coef = T(0.0)) :
      coefs(arma::Col<T>{coef}),
      wavenumbers(arma::zeros<arma::Mat<T>>(dim, 1)) {}

  explicit
  inline Exponential(const exponential::Term<T> & term) :
      coefs(arma::Col<T>{term.coef}),
      wavenumbers(arma::conv_to<arma::Mat<T>>::from(term.wavenumbers)) {}

  explicit
  inline Exponential(const arma::Col<T> & coefs, const arma::Mat<T> wavenumbers)
      :
      coefs(coefs),
      wavenumbers(wavenumbers) {
    if (coefs.n_elem != wavenumbers.n_cols) {
      throw Error("Different number of terms between coefs and wavenumbers");
    }
  }

  template<typename U>
  std::common_type_t<T, U> at(const arma::Col<T> & position) {
    if (position.n_eleme != wavenumbers.n_rows) {
      throw Error("different dimension between position and exponential term");
    }

    arma::Mat<std::common_type_t<T, U>> duplicated_position
        = arma::zeros<arma::Mat<std::common_type_t<T, U>>>(
            arma::size(wavenumbers));

    duplicated_position.each_col() += position;

    return arma::prod(
        this->coefs %
        arma::prod(arma::exp(this->wavenumbers % duplicated_position)).t()
    );
  }

  inline
  exponential::Term<T> term(arma::uword index) const {
    if (index >= this->coefs.n_elem) {
      return ("The specified exponential term does not exist");
    }

    return exponential::Term<T>(this->coefs(index),
                                this->wavenumbers.col(index));
  }

  inline
  arma::uword dim() const {
    return this->indices.n_rows;
  }

  inline
  Exponential<T> derivative(const arma::uword index) const {
    if (index >= this->dim()) {
      throw Error("Derivative operator out of bound");
    }
    Exponential<T> result = Exponential<T>(this->term(0).derivative(index));

#pragma omp parallel for
    for (arma::uword i = 0; i < this->coefs.n_elem; i++) {
      result = result + this->term(i).derivative(index);
    }

    return result;
  }

  template<typename U>
  Exponential<std::common_type_t<T, U>>
  operator+(const Exponential<U> & B) const {
    if (this->dim() != B.dim()) {
      throw Error("Different dimension between added exponential terms");
    }
    const arma::Col<std::common_type_t<T, U>>
        new_this_coefs = arma::conv_to<arma::Col<std::common_type_t<T, U>>>::from(
        this->coefs);
    const arma::Col<std::common_type_t<T, U>>
        new_B_coefs = arma::conv_to<arma::Col<std::common_type_t<T, U>>>::from(
        B.coefs);
    const arma::Col<std::common_type_t<T, U>>
        new_this_wavenumbers = arma::conv_to<arma::Col<std::common_type_t<T, U>>>::from(
        this->wavenumbers);
    const arma::Col<std::common_type_t<T, U>>
        new_B_wavenumbers = arma::conv_to<arma::Col<std::common_type_t<T, U>>>::from(
        B.wavenumbers);

    return {arma::join_cols(new_this_coefs, new_B_coefs),
            arma::join_rows(new_this_wavenumbers, new_B_wavenumbers)};
  }

  template<typename U>
  Exponential<std::common_type_t<T, U>>
  operator+(const exponential::Term<U> & B) const {

    return *this + Exponential(B);
  }

  template<typename U>
  Exponential<std::common_type_t<T, U>> operator+(const U B) const {

    return *this + Exponential(this->dim(), std::common_type_t<T, U>(B));
  }

  template<typename U>
  Exponential<std::common_type_t<T, U>>
  operator*(const exponential::Term<U> & B) const {
    if (this->dim() != B.dim()) {
      throw Error("Different dimension between added exponential terms");
    }
    auto new_wavenumbers = arma::conv_to<arma::Col<std::common_type_t<T, U>>>::from(
        this->wavenumbers);
    new_wavenumbers.each_col() += B.wavenumbers;
    const arma::Col<std::common_type_t<T, U>>
        new_coefs = this->coefs * B.coef;

    return {new_coefs, new_wavenumbers};
  }

  template<typename U>
  Exponential<std::common_type_t<T, U>>
  operator*(const Exponential<U> & B) const {
    Exponential<std::common_type_t<T, U>> result_0 = (*this) * B.term(0);

#pragma omp parallel for
    for (arma::uword i = 1; i < B.coefs.n_elem; i++) {
      result_0 = result_0 + (*this) * B.term(i);
    }

    return result_0;
  }

  template<typename U>
  Exponential<std::common_type_t<T, U>> operator*(const U B) const {
    return {this->coefs * B, this->wavenumbers};
  }

  template<typename U>
  Exponential<std::common_type_t<T, U>>
  operator-(const Exponential<U> & B) const {
    return *this + B * (-1.0);
  }

  template<typename U>
  Exponential<std::common_type_t<T, U>> operator-(const U B) const {
    return *this + (-B);
  }

  template<typename U>
  Exponential<std::common_type_t<T, U>> operator/(const U B) const {
    return *(this) * (1.0 / B);
  }

  template<typename U>
  Exponential<std::common_type_t<T, U>>
  operator/(const exponential::Term<T> & B) const {
    auto new_wavenumbers = arma::conv_to<arma::Mat<std::common_type_t<T, U>>>::from(
        this->wavenumbers);
    new_wavenumbers.each_col() -= B.wavenumbers;
    const arma::Col<std::common_type_t<T, U>> new_coefs = this->coefs / B.coef;

    return {new_coefs, new_wavenumbers};
  }
};

}
}
#endif //MATH_EXPONENTIAL_H
