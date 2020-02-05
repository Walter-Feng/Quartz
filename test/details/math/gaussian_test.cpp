#include <Catch2/catch.hpp>

#include <quartz>

namespace quartz {

using namespace quartz::math;

TEST_CASE("Gaussian Value") {
  SECTION("One Dimension") {
    const arma::vec grid = arma::linspace(-5, 5, 10);

    const Gaussian gaussian1 = Gaussian(1, 1.0);

    for (arma::uword i = 0; i < grid.n_elem; i++) {
      CHECK(std::real(gaussian1.at(arma::vec{grid(i)})) == 1.0);
    }

    const Gaussian gaussian2 = Gaussian(arma::eye<arma::mat>(1, 1));

    for (arma::uword i = 0; i < grid.n_elem; i++) {
      CHECK(std::real(gaussian2.at(arma::vec{grid(i)})) ==
            std::exp(-0.5 * grid(i) * grid(i)));
    }

    const Gaussian gaussian3 = Gaussian(arma::eye<arma::mat>(1, 1), arma::vec{2.},2.0);

    for (arma::uword i = 0; i < grid.n_elem; i++) {
      CHECK(std::real(gaussian3.at(arma::vec{grid(i)})) ==
            2.0 * std::exp(-0.5 * grid(i) * grid(i) + 2. * grid(i)));
    }
  }

  SECTION("Multi Dimension") {
    const arma::vec grid = arma::linspace(-5, 5, 10);

    const Gaussian gaussian1 = Gaussian(1, 1.0);

    for (arma::uword i = 0; i < grid.n_elem; i++) {
      CHECK(std::real(gaussian1.at(arma::vec{grid(i)})) == 1.0);
    }

    const Gaussian gaussian2 = Gaussian(arma::eye<arma::mat>(1, 1));

    for (arma::uword i = 0; i < grid.n_elem; i++) {
      CHECK(std::real(gaussian2.at(arma::vec{grid(i)})) ==
            std::exp(-0.5 * grid(i) * grid(i)));
    }

    const Gaussian gaussian3 = Gaussian(arma::eye<arma::mat>(1, 1), arma::vec{2.},2.0);

    for (arma::uword i = 0; i < grid.n_elem; i++) {
      CHECK(std::real(gaussian3.at(arma::vec{grid(i)})) ==
            2.0 * std::exp(-0.5 * grid(i) * grid(i) + 2. * grid(i)));
    }
  }
}


}
