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

    const Gaussian gaussian3 = Gaussian(arma::eye<arma::mat>(1, 1),
                                        arma::vec{2.}, 2.0);

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

    const Gaussian gaussian3 = Gaussian(arma::eye<arma::mat>(1, 1),
                                        arma::vec{2.}, 2.0);

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
  CHECK(
      std::abs((test * test).integral() - wigner_transform.integral()) < 1e-14);

  const Gaussian test_2 = Gaussian(arma::mat{2});
  const Gaussian wigner_transform_2 = test_2.wigner_transform();

  CHECK(test_2.integral() == std::sqrt(pi));
  CHECK(std::abs((test_2 * test_2).integral() - wigner_transform_2.integral()) <
        1e-14);

  const Gaussian with_monomial = Gaussian(arma::mat{1}, arma::vec{2});
  const Gaussian wigner_transform_monomial = with_monomial.wigner_transform();
  CHECK(with_monomial.integral() == std::sqrt(2. * pi) * std::exp(2));
  CHECK((with_monomial * with_monomial).integral() ==
        wigner_transform_monomial.integral());


  const Gaussian test_multi_1 = Gaussian(arma::mat{{3, 0},
                                                   {0, 2}}, arma::vec{1, 2});
  const Gaussian wigner_transform_multi_1 = test_multi_1.wigner_transform();
  CHECK(std::abs(test_multi_1.integral() -
                 std::sqrt(2.0 / 3.0) * std::exp(7.0 / 6.0) * pi) < 1e-14);
  CHECK(std::abs((test_multi_1 * test_multi_1).integral()
                 - wigner_transform_multi_1.integral()) < 1e-14);

  const Gaussian test_multi = Gaussian(arma::mat{{6, 1},
                                                 {1, 4}}, arma::vec{1, 2});
  const Gaussian wigner_transform_multi = test_multi.wigner_transform();
  CHECK((test_multi * test_multi).integral() ==
        wigner_transform_multi.integral());
}


}
