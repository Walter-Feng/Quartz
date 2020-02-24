#include <catch.hpp>

#include <quartz>

namespace quartz {

using namespace quartz::math;
using namespace quartz::method;

template<typename T>
arma::cx_mat test_schrotinger_wrapper(const arma::Mat<T> & result,
                                 const double dt) {
  if (!result.is_square()) {
    throw Error("The matrix being wrapped for propagation is not square");
  }
  return
      arma::inv(arma::eye(arma::size(result)) +
                0.5 * cx_double{0.0, 1.0} * dt * result) *
      (arma::eye(arma::size(result)) -
       0.5 * cx_double{0.0, 1.0} * dt * result);
}

TEST_CASE("Schrotinger wrapper") {


  const arma::mat unit = arma::randu(10, 10);

  const arma::mat symmetric = unit + unit.t();

  const double dt = 0.01;

  const dvr::Operator test = dvr::Operator(symmetric);

  const dvr::State random_state =
      dvr::State(Gaussian(arma::mat{1.}, arma::vec{1}), arma::uvec{10},
                 arma::mat{{-5, 5}});

  const arma::cx_mat wrapped = test_schrotinger_wrapper(symmetric, dt);

  const auto propagator =
      math::schrotinger_wrapper<dvr::Operator, dvr::State, Polynomial<double>>(
          test, Polynomial<double>(1));

  CHECK(arma::approx_equal(propagator(random_state, dt).coefs,
                           wrapped * random_state.coefs, "abs_diff", 1e-10));

}

}