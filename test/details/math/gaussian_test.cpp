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

TEST_CASE("Gaussian Integral") {
  const Gaussian test = Gaussian(arma::mat{1});
  const Gaussian wigner_transform = test.wigner_transform();

  CHECK(test.integral() == std::sqrt(2. * pi));
  CHECK(std::abs((test * test).integral() - wigner_transform.integral()) < 1e-14);

  const Gaussian test_2 = Gaussian(arma::mat{2});
  const Gaussian wigner_transform_2 = test_2.wigner_transform();

  CHECK(test_2.integral() == std::sqrt(pi));
  CHECK(std::abs((test_2 * test_2).integral() - wigner_transform_2.integral()) < 1e-14);

  const Gaussian with_monomial = Gaussian(arma::mat{1},arma::vec{2});
  const Gaussian wigner_transform_monomial = with_monomial.wigner_transform();
  CHECK(with_monomial.integral() == std::sqrt(2. * pi) * std::exp(2));
  CHECK((with_monomial * with_monomial).integral() == wigner_transform_monomial.integral());
}



}
