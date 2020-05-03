#ifndef MATH_POLYNOMIAL_H
#define MATH_POLYNOMIAL_H

#include "alias.h"
#include "trivial.h"
#include "quartz_internal/error.h"
#include "quartz_internal/util/member_function_wrapper.h"
#include "quartz_internal/util/type_converter.h"
#include "quartz_internal/details/math/space.h"

namespace math {
namespace polynomial {

// Term represents the each term a polynomial will have.
template<typename T>
struct Term {
  T coef;
  lvec exponents;

  template<typename U>
  std::common_type_t<T, U> at(const arma::Col<U> & position) const {

    if (position.n_elem != this->exponents.n_elem) {
      throw Error(
          "Different dimension between the position and polynomial term");
    }

    auto result = std::common_type_t<T, U>(1.0);

    for (arma::uword i = 0; i < position.n_elem; i++) {
      if (this->exponents(i) == 0)
        continue;

      result *= std::pow(position(i), this->exponents(i));
    }

    return this->coef * result;
  }

  inline
  Term() :
      coef(0.0),
      exponents() {}

  explicit
  inline
  Term(const arma::uword dim, const T coef = T{0.0}) :
      coef(coef),
      exponents(arma::zeros<lvec>(dim)) {}

  inline
  Term(const T coef, const lvec & indices) :
      coef(coef),
      exponents(indices) {}

  inline
  Term(const T coef, const arma::uvec & indices) :
      coef(coef),
      exponents(arma::conv_to<lvec>::from(indices)) {}

  arma::uword dim() const {
    return this->exponents.n_elem;
  }

  template<typename U>
  Term<std::common_type_t<T, U>>
  scale(const arma::Col<U> & scaling) const {
    return {this->at(scaling), this->exponents};
  }

  template<typename U>
  bool is_same_term(const Term<U> & term) const {
    if (this->exponents == term.exponents) return true;

    return false;
  }

  inline
  Term<T> derivative(const arma::uword index) const {
    if (this->exponents(index) == 0) {
      return {T{0.}, arma::zeros<lvec>(this->dim())};
    } else {
      lvec new_indices = this->exponents;
      new_indices(index) -= 1;
      return {this->coef * (double) (new_indices(index) + 1), new_indices};
    }
  }

  inline
  Term<T> derivative(const arma::uvec & index) const {
    if (index.n_elem != this->exponents.n_elem) {
      throw Error("Derivative operator out of bound");
    }

    Term<T> result = *this;
#pragma omp parallel for
    for (arma::uword i = 0; i < index.n_elem; i++) {
      for (arma::uword j = 0; j < index(i); j++) {
        result = result.derivative(i);
      }
    }

    return result;
  }

  template<typename U>
  auto differentiate(const U & function) const {
    if (arma::min(this->exponents) < 0) {
      throw Error("Quartz does not support integration operator");
    }
    return quartz::derivative(function, arma::conv_to<arma::uvec>::from(
        this->exponents)) * this->coef;
  }

  inline
  Term<T> pow(const arma::uword power) const {
    return {std::pow(this->coef, power), this->exponents * power};
  }


  bool operator==(const Term<T> & term) const {
    return this->coef == term.coef && this->is_same_term(term);
  }

  Term& operator=(const Term &) = default;
};

} // namespace polynomial

// The Polynomial struct is stored as a list of exponents and the corresponding
// coefficients. The exponents are stored column-wise.

template<typename T>
struct Polynomial {
public:

  arma::Col<T> coefs;
  lmat exponents;

  inline
  Polynomial(void) :
      coefs(),
      exponents() {}

  inline
  Polynomial(const arma::Col<T> & coefs, const lmat & exponents) :
      coefs(coefs),
      exponents(exponents) {
    if (coefs.n_elem != exponents.n_cols) {
      throw Error(
          "the number between coefficients and the exponents is not consistent");
    }
  }

  inline
  Polynomial(const polynomial::Term<T> & term) :
      coefs(arma::Col<T>{term.coef}),
      exponents(lmat(term.exponents)) {}

  inline
  Polynomial(const Polynomial<T> & term) :
      coefs(term.coefs),
      exponents(term.exponents) {}

  inline
  Polynomial(const arma::uword dim, const T coef = 0.0) :
      coefs(arma::Col<T>{coef}),
      exponents(arma::zeros<lmat>(dim, 1)) {}

  inline
  polynomial::Term<T> term(arma::uword index) const {
    if (index >= this->coefs.n_elem) {
      throw Error("The specified polynomial term does not exist");
    }
    return polynomial::Term<T>{this->coefs(index), this->exponents.col(index)};
  }

  inline
  arma::uword dim() const {
    return this->exponents.n_rows;
  }

  inline
  long long grade() const {
    return arma::max(arma::sum(this->exponents));

  }

  inline
  Polynomial<double> real() const {
    return Polynomial<double>(arma::real(this->coefs), this->exponents).clean();
  }

  inline
  Polynomial<double> imag() const {
    return Polynomial<double>(arma::imag(this->coefs), this->exponents).clean();
  }

  inline
  Polynomial<double> abs() const {
    return Polynomial<double>(arma::abs(this->coefs), this->exponents).clean();
  }

  inline
  Polynomial<T> conj() const {
    if constexpr(std::is_same<T,double>::value) {
      return *this;
    } else {
      const arma::cx_vec new_coefs = arma::conj(this->coefs);

      return Polynomial<T>(new_coefs, this->exponents);
    }
  }
  template<typename U>
  std::common_type_t<T, U> at(const arma::Col<U> & position) const {

    if (position.n_elem != this->exponents.n_rows) {
      throw Error(
          "Different dimension between the position and polynomial term");
    };

    auto result = std::common_type_t<T, U>(0.0);

    for (arma::uword i = 0; i < this->exponents.n_cols; i++) {
      const polynomial::Term<T> term = this->term(i);
      result += term.at(position);
    }

    return result;
  }

  inline
  Polynomial<T> derivative(const arma::uword index) const {
    if (index >= this->dim()) {
      throw Error("Derivative operator out of bound");
    }
    Polynomial<T> result = Polynomial<T>(this->term(0).derivative(index));

    for (arma::uword i = 1; i < this->coefs.n_elem; i++) {
      result = result + this->term(i).derivative(index);
    }

    return result.clean();
  }

  inline
  Polynomial<T> derivative(const arma::uvec & index) const {
    if (index.n_elem != this->dim()) {
      throw Error("Derivative operator out of bound");
    }
    Polynomial<T> result = *this;

#pragma omp parallel for
    for (arma::uword i = 0; i < index.n_elem; i++) {
      for (arma::uword j = 0; j < index(i); j++) {
        result = result.derivative(i);
      }
    }

    return result.clean();
  }

  template<typename U>
  auto differentiate(const U & function) const {
    auto result = this->term(0).differentiate(function);
#pragma omp parallel for
    for (arma::uword i = 0; i < this->coefs.n_elem; i++) {
      const polynomial::Term<T> term = this->term(i);
      result = result + term.differentiate(function);
    }

    return result;
  }

  template<typename U>
  Polynomial<std::common_type_t<T, U>>
  displace(const arma::Col<U> & displacement) const {
    if (this->dim() != displacement.n_elem) {
      throw Error(
          "Different dimension between the displacement and polynomial term");
    }

    const auto dim = this->dim();

    auto result =
        Polynomial<std::common_type_t<T, U>>(dim);

    const auto binomial =
        [](const double n, const double i) -> double {
          return math::factorial(n) / factorial(i) / factorial(n - i);
        };

    const auto term_displace =
        [&binomial](const polynomial::Term<T> & term,
                    const arma::Col<U> & displacement)
            -> Polynomial<std::common_type_t<T, U>> {

          const arma::uword dim = term.dim();
          const auto & exponent = term.exponents;
          const arma::uvec grid = arma::conv_to<arma::uvec>::from(exponent + 1);
          const auto iterations = space::auto_iteration_over_dims(grid);

          auto result = Polynomial<std::common_type_t<T, U>>(dim);

#pragma omp parallel for
          for (arma::uword i = 0; i < iterations.n_cols; i++) {
            const lvec displacements_poly = arma::conv_to<lvec>::from(
                iterations.col(i));
            const lvec new_exponents = exponent - displacements_poly;

            const math::polynomial::Term<double> local_term(1.0,
                                                            displacements_poly);

            double binomial_coef = 1;
            for (arma::uword j = 0; j < dim; j++) {
              binomial_coef *= binomial(exponent(j), displacements_poly(j));
            }

            result = result + math::polynomial::Term<double>(
                term.coef * binomial_coef * local_term.at(displacement),
                new_exponents);
          }

          return result;
        };

#pragma omp parallel for
    for (arma::uword i = 0; i < this->coefs.n_elem; i++) {
      result = result + term_displace(this->term(i), displacement);
    }

    return result;
  }

  Polynomial<T> scale(const arma::vec & scaling) const {
    auto result = Polynomial<T>(this->term(0).scale(scaling));

    for (arma::uword i = 1; i < this->coefs.n_elem; i++) {
      result = result + this->term(i).scale(scaling);
    }

    return result;
  }

  Polynomial<T> pow(const arma::uword power) const {
    if (power == 0) {
      return Polynomial<T>(this->dim(), 1.0);
    }
    Polynomial<T> result = *this;
    for (arma::uword i = 1; i < power; i++) {
      result = result * *this;
    }

    return result;
  }

  template<typename U>
  Polynomial<std::common_type_t<T, U>>
  operator()(const std::vector<Polynomial<U>> & polynomial_list) const {

    const auto dim = this->dim();

    if (this->dim() != polynomial_list.size()) {
      throw Error("Mismatched number between the operator and term");
    }

    const auto term_operate = [dim](const polynomial::Term<T> & term,
                                    const std::vector<Polynomial<U>> & polynomial_list)
        -> Polynomial<std::common_type_t<T, U>> {

      auto result = Polynomial<std::common_type_t<T, U>>(
          polynomial_list[0].dim(), 1.0);
#pragma omp parallel for
      for (arma::uword i = 0; i < dim; i++) {
        result = result * polynomial_list[i].pow(term.exponents(i));
      }

      return result * term.coef;
    };

    auto result = Polynomial<std::common_type_t<T, U>>(polynomial_list[0].dim(),
                                                       0.0);
    for (arma::uword i = 0; i < this->coefs.n_elem; i++) {
      result = result + term_operate(this->term(i), polynomial_list);
    }

    return result;
  }

  template<typename U>
  Polynomial<std::common_type_t<T, U>>
  operator+(const Polynomial<U> & B) const {
    const lmat new_indices = arma::join_rows(this->exponents, B.exponents);
    const auto converted_this_coefs =
        arma::conv_to<arma::Col<std::common_type_t<T, U>>>::from(this->coefs);
    const auto converted_B_coefs =
        arma::conv_to<arma::Col<std::common_type_t<T, U>>>::from(B.coefs);
    const arma::Col<std::common_type_t<T, U>>
        new_coefs =
        arma::join_cols(converted_this_coefs, converted_B_coefs);

    return Polynomial<std::common_type_t<T, U>>{new_coefs, new_indices}.clean();
  }

  template<typename U>
  Polynomial<std::common_type_t<T, U>> operator+(const U B) const {

    const lvec dummy_indices = arma::zeros<lvec>(
        this->exponents.n_rows);
    const lmat new_indices = arma::join_rows(this->exponents,
                                             dummy_indices);
    const arma::Col<std::common_type_t<T, U>> converted_coefs =
        arma::conv_to<arma::Col<std::common_type_t<T, U>>>::from(this->coefs);
    const arma::Col<std::common_type_t<T, U>> new_coefs = arma::join_cols(
        converted_coefs, arma::Col<std::common_type_t<T, U>>{B});

    return Polynomial<std::common_type_t<T, U>>{new_coefs, new_indices}.clean();
  }

  template<typename U>
  Polynomial<std::common_type_t<T, U>>
  operator+(const polynomial::Term<U> & B) const {
    const lmat new_indices = arma::join_rows(this->exponents, B.exponents);
    const auto converted_this_coefs =
        arma::conv_to<arma::Col<std::common_type_t<T, U>>>::from(this->coefs);
    const auto converted_B_coef = arma::Col<std::common_type_t<T, U>>{B.coef};
    const arma::Col<std::common_type_t<T, U>>
        new_coefs = arma::join_cols(converted_this_coefs, converted_B_coef);

    return Polynomial<std::common_type_t<T, U>>{new_coefs, new_indices}.clean();
  }

  template<typename U>
  Polynomial<std::common_type_t<T, U>>
  operator*(const polynomial::Term<U> & B) const {
    lmat new_indices = this->exponents;
    new_indices.each_col() += B.exponents;
    const arma::Col<std::common_type_t<T, U>>
        new_coefs = this->coefs * B.coef;

    return Polynomial<std::common_type_t<T, U>>{new_coefs, new_indices}.clean();
  }

  template<typename U>
  Polynomial<std::common_type_t<T, U>>
  operator*(const Polynomial<U> & B) const {
    Polynomial<std::common_type_t<T, U>> result_0 = (*this) * B.term(0);

    for (arma::uword i = 1; i < B.coefs.n_elem; i++) {
      result_0 = result_0 + (*this) * B.term(i);
    }

    return result_0.clean();
  }

  template<typename U>
  Polynomial<std::common_type_t<T, U>> operator*(const U B) const {
    return Polynomial<std::common_type_t<T, U>>{this->coefs * B,
                                                this->exponents}.clean();
  }

  template<typename U>
  Polynomial<std::common_type_t<T, U>>
  operator-(const Polynomial<U> & B) const {
    return *this + B * (-1.0);
  }

  template<typename U>
  Polynomial<std::common_type_t<T, U>> operator-(const U B) const {
    return *this + (-B);
  }

  template<typename U>
  Polynomial<std::common_type_t<T, U>> operator/(const U B) const {
    return *(this) * (1.0 / B);
  }

  template<typename U>
  Polynomial<std::common_type_t<T, U>>
  operator/(const polynomial::Term<T> & B) const {
    lmat new_indices = this->exponents;
    new_indices.each_col() -= B.exponents;
    const arma::Col<std::common_type_t<T, U>> new_coefs = this->coefs / B.coef;

    return Polynomial<std::common_type_t<T, U>>{new_coefs, new_indices}.clean();
  }

  Polynomial<T> sort() const {
    const lvec maximum_exponents = arma::max(this->exponents, 1);
    const lvec minimum_exponents = arma::min(this->exponents, 1);

    const arma::uvec grid =
        arma::conv_to<arma::uvec>::from(maximum_exponents - minimum_exponents)
        + 1;

    const auto table = math::space::grids_to_table(grid);

    lmat indices = this->exponents;
    indices.each_col() -= minimum_exponents;
    const arma::umat converted_indices = arma::conv_to<arma::umat>::from(
        indices);

    arma::uvec key(converted_indices.n_cols);


#pragma omp parallel for
    for (arma::uword i = 0; i < converted_indices.n_cols; i++) {
      const arma::uvec index = converted_indices.col(i);

      key(i) = math::space::indices_to_index(index, table);
    }

    const arma::uvec unique_elements = arma::unique(key);

    arma::Col<T> result_coefs(unique_elements.n_elem);
    lmat result_exponents(this->dim(), unique_elements.n_elem);

    for (arma::uword i = 0; i < unique_elements.n_elem; i++) {
      const arma::uvec
          identical_terms_indices = arma::find(key == unique_elements(i));

      const lvec corresponding_exponent =
          this->exponents.col(identical_terms_indices(0));

      const arma::Col<T> corresponding_coef =
          this->coefs.rows(identical_terms_indices);

      result_coefs(i) = arma::sum(corresponding_coef);

      result_exponents.col(i) = corresponding_exponent;
    }

    return Polynomial<T>(result_coefs, result_exponents);

  }

  Polynomial<T> clean() const {

    const auto sorted_polynomial = this->sort();
    const arma::uvec non_zero = arma::find(sorted_polynomial.coefs);

    if (non_zero.n_elem == 0) {
      return Polynomial<T>(sorted_polynomial.dim());
    }
    return Polynomial<T>(sorted_polynomial.coefs.rows(non_zero),
                         sorted_polynomial.exponents.cols(non_zero));
  }

  std::string to_string(const int precision = 3,
                        const int width = -1) const {

    const auto printer = [](const Polynomial<double> term,
                            const int precision,
                            const int width) {

      const std::vector<std::string> variables =
          util::variable_names(term.dim());

      std::string result = " ";

      if (width <= 0) {
        result += fmt::format("{:.{}}", term.coefs(0), precision);
      } else {
        result += format(term.coefs(0), precision, width);
      }
      for (arma::uword j = 0; j < term.exponents.n_rows; j++) {
        result =
            result + variables[j] + "^" + std::to_string(term.exponents(j, 0)) +
            " ";
      }

      for (arma::uword i = 1; i < term.exponents.n_cols; i++) {

        if (term.coefs(i) < 0) {
          result += "- ";
        } else {
          result += "+ ";
        }

        if (width <= 0) {
          result += fmt::format("{:.{}}", std::abs(term.coefs(i)), precision);
        } else {
          result += format(term.coefs(i), precision, width);
        }
        for (arma::uword j = 0; j < term.exponents.n_rows; j++) {
          result =
              result + variables[j] + "^" +
              std::to_string(term.exponents(j, i)) + " ";
        }

      }
      return result;

    };

    if constexpr(std::is_same<T, cx_double>::value) {
      return " (" + printer(this->real(), precision, width)
             + "," + printer(this->imag(), precision, width) + ") ";
    } else {
      return printer(*this, precision, width);
    }

  }

  Polynomial<T>& operator=(const Polynomial<T> &) = default;
};

template<typename T>
std::vector<Polynomial<T>> transform(const arma::Mat<T> & transform_matrix) {
  std::vector<Polynomial<T>> result[transform_matrix.n_cols];

#pragma omp parallel for
  for (arma::uword i = 0; i < transform_matrix.n_cols; i++) {
    result[i] = Polynomial<T>(transform_matrix.row(i).st(),
                              arma::eye<lmat>(arma::size(transform_matrix)));
  }

  return result;
}
}


#endif //MATH_POLYNOMIAL_H
