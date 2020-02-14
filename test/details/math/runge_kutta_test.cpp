#include <Catch2/catch.hpp>

#include <quartz>

namespace quartz {

using namespace quartz::math;

TEST_CASE("Runge Kutta Method") {
SECTION("Time Independent Version") {
  struct unit {
    double core = 1;

    PropagationType propagation_type() const {
      return Classic;
    }

    arma::vec operator*(const arma::vec & B) const {
      return this->core * B;
    }
  } const_operator;

  struct dummy_potential {

  } dummy;
  const arma::vec state = arma::ones(10);

  const auto propagator =
      runge_kutta_2<unit, arma::vec, dummy_potential>(const_operator,dummy);

  CHECK(arma::approx_equal(propagator(state,0.01),state + 0.01005, "abs_diff", 1e-16));

  const auto propagator2 =
      runge_kutta_4<unit, arma::vec, dummy_potential>(const_operator,dummy);
  
  CHECK(arma::approx_equal(propagator2(state,0.1),state + 0.105170833333, "abs_diff", 1e-8));
}

}
}