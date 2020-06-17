#include "exponential.h"

#include <quartz>

#include "src/util/ptree.h"

namespace quartz {
namespace parse {

namespace ptree = boost::property_tree;

math::Exponential<double> exponential(const ptree::ptree & input) {

  const arma::mat wavenumbers = util::get_mat<double>(
      input.get_child("wavenumbers"));

  if(input.get_child_optional("coefs")) {
    const arma::vec coefs = arma::vec(util::get_list<double>(input.get_child("coefs")));

    return math::Exponential<double>(coefs, wavenumbers);
  }

  else {
    const arma::vec coefs = arma::ones<arma::vec>(wavenumbers.n_rows);

    return math::Exponential<double>(coefs, wavenumbers);
  }
}

}
}