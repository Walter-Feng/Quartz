#include "gaussian.h"

#include <quartz>

#include "src/util/ptree.h"

namespace quartz {
namespace parse {

namespace ptree = boost::property_tree;

math::Gaussian<cx_double> gaussian(const ptree::ptree & input) {

  const arma::mat covariance = util::get_mat<double>(
      input.get_child("covariance"));

  const double coef = input.get<double>("coef", 1.0);

  arma::vec mean = arma::zeros<arma::vec>(covariance.n_rows);
  arma::vec phase_factor = arma::zeros<arma::vec>(covariance.n_rows);

  if(input.get_child_optional("mean")) {
    mean = arma::vec(util::get_list<double>(input.get_child("mean")));
  }
  if(input.get_child_optional("phase_factor")) {
    phase_factor = arma::vec(util::get_list<double>(input.get_child("phase_factor")));
  }

  return math::Gaussian<double>(covariance, mean, coef).with_phase_factor(phase_factor);
}

}
}