#include "m_dvr.h"

#include <quartz>
#include <boost/property_tree/ptree.hpp>

#include "src/util/ptree.h"
#include "src/parse/printer.h"

#include "src/parse/math/math_object.h"
#include "src/parse/math/gaussian.h"

namespace quartz {

namespace ptree = boost::property_tree;

ptree::ptree m_dvr(const ptree::ptree & input) {

  const std::function<math::Gaussian<cx_double>(const ptree::ptree &)>
      parse_initial = [](const ptree::ptree & pt) -> math::Gaussian<cx_double> {
    return parse::gaussian(pt);
  };

  const std::function<MathObject<double>(const ptree::ptree &)>
    parse_potential = [](const ptree::ptree & pt)
      ->MathObject<double> {return parse::math_object(pt);};

  std::vector<math::Gaussian<cx_double>> initial =
      util::get_list(input.get_child("initial"), parse_initial);

  const arma::field<MathObject<double>> potentials =
      util::get_mat_object(input.get_child("potential"), parse_potential);

  const arma::uvec grid =
      arma::uvec(util::get_list<arma::uword>(input.get_child("grid")));

  const arma::mat range =
      util::get_mat<double>(input.get_child("range")).t();

  arma::vec masses = arma::ones(grid.n_elem);

  const auto steps = input.get<arma::uword>("steps");
  const auto dt = input.get<double>("dt");

  if (input.get_child_optional("mass")) {
    masses = arma::vec(util::get_list<double>(input.get_child("mass")));
  }

  method::m_dvr::State initial_state(initial, grid, range, masses);
  method::m_dvr::Operator op(initial_state, potentials);
  auto wrapper =
      math::schrotinger_wrapper<method::m_dvr::Operator,
                                method::m_dvr::State,
                                arma::field<MathObject<double>>>;

  ptree::ptree result;

  auto printer_pair = printer(input, result, initial_state);

  propagate(initial_state, op, wrapper, potentials, printer_pair.first, steps,
            dt, printer_pair.second);

  return result;
}

}