#include <catch.hpp>

#include <quartz>

namespace quartz {
using namespace quartz::math;
using namespace quartz::method;

TEST_CASE("DVR Test") {
  SECTION("One Dimension") {

    const dvr::State random_state =
        dvr::State(Gaussian<double>(arma::mat{1.},arma::vec{1}),arma::uvec{100},arma::mat{{-10,10}});

    const dvr::Operator op = dvr::Operator(random_state, Polynomial(arma::vec{0.5},lvec{2}));

    CHECK(std::abs(random_state.positional_expectation()(0) - 1.) < 1e-14);
    CHECK(std::abs(random_state.momentum_expectation()(0) - 0.) < 1e-14);

  }
}
}