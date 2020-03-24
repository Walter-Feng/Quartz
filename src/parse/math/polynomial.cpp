#include "polynomial.h"

#include "src/util/ptree.h"

namespace quartz {
namespace parse {

math::Polynomial<double> polynomial(const ptree::ptree & input) {

  const lmat exponents = util::get_mat<long long>(input.get_child("exponents"));

  if(input.get_child_optional("coefs")) {
    const arma::vec coefs = arma::vec(util::get_list<double>(input.get_child("coefs")));

    return math::Polynomial<double>(coefs, exponents);
  }

  else {
    const arma::vec coefs = arma::ones<arma::vec>(exponents.n_rows);

    return math::Polynomial<double>(coefs, exponents);
  }

  __builtin_unreachable;
}

}
}