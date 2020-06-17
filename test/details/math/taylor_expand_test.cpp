#include <catch.hpp>

#include <quartz>

namespace quartz {

using namespace quartz::math;

TEST_CASE("Taylor Expansion") {
  SECTION("indices_with_same_sum") {
    CHECK(
        arma::approx_equal(math::details::indices_with_same_sum(5, 1),
                           arma::eye<arma::umat>(5, 5), "abs_diff", 0)
    );
  }

  SECTION("taylor expansion") {
    const math::Polynomial<double> quartic(arma::vec{1.0}, lmat{4});

    const auto expanded = math::taylor_expand<math::Polynomial<double>, double>(quartic, 5);

    CHECK(expanded(arma::vec{3.0},arma::vec{0.0}) == quartic.at(arma::vec{3.0}));

    const math::GaussianWithPoly<double> gaussian(-arma::mat{1}, arma::vec{0.0});

    math::taylor_expand<math::GaussianWithPoly<double>, double>(gaussian,10);

    math::taylor_expand<math::GaussianWithPoly<double>, double>(gaussian, arma::vec{0.0}, 10);
  }
}

}
