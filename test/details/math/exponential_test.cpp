#include <catch.hpp>

#include <quartz>

namespace quartz {

using namespace quartz::math;

TEST_CASE("Exponential Value") {
  SECTION("One Dimension") {
    const arma::vec grid = arma::linspace(-5, 5, 10);
    const arma::cx_vec cx_grid = arma::cx_vec{arma::linspace(-5, 5, 10),
                                              arma::linspace(5, -5, 10)};

    const Exponential<double> exponential = Exponential<double>(arma::vec{1, 2},
                                                                arma::mat{{2, 4}});

    for (arma::uword i = 0; i < grid.n_elem; i++) {
      CHECK(exponential.at(arma::vec{grid(i)})
            == std::exp(2 * grid(i)) + 2 * std::exp(4 * grid(i)));
    }

    for (arma::uword i = 0; i < grid.n_elem; i++) {
      const auto lhs = exponential.at(arma::cx_vec{cx_grid(i)});
      const auto rhs =
          std::exp(2. * cx_grid(i)) + 2. * std::exp(4. * cx_grid(i));
      CHECK(lhs == rhs);
    }
  }

  SECTION("Multi Dimension") {
    const arma::mat grid = arma::randu<arma::mat>(10, 10);

    const arma::vec wavenumbers = arma::randu<arma::vec>(10);

    const auto term = exponential::Term<double>(1, wavenumbers);

    const Exponential<double> exponential = Exponential<double>(term);

    arma::vec rhs = arma::vec(10);

    for (arma::uword i = 0; i < 10; i++) {
      rhs(i) = std::exp(arma::sum(grid.col(i) % wavenumbers));
    }

    CHECK(arma::approx_equal(at(exponential, grid), rhs, "abs_diff", 1e-14));
  }
}

TEST_CASE("Exponential Derivative") {
  SECTION("One Dimension") {
    const arma::vec grid = arma::linspace(-5, 5, 10);
    const arma::cx_vec cx_grid = arma::cx_vec{arma::linspace(-5, 5, 10),
                                              arma::linspace(5, -5, 10)};

    const Exponential<double> exponential = Exponential<double>(arma::vec{1, 2},
                                                                arma::mat{{2, 4}});

    const auto derivative = exponential.derivative(0);

    for (arma::uword i = 0; i < grid.n_elem; i++) {
      CHECK(derivative.at(arma::vec{grid(i)})
            == 2. * std::exp(2 * grid(i)) + 8. * std::exp(4 * grid(i)));
    }
  }

  SECTION("Multi Dimension") {
    const arma::mat grid = arma::randu<arma::mat>(10, 10);

    const arma::vec wavenumbers = arma::randu<arma::vec>(10);

    const auto term = exponential::Term<double>(1, wavenumbers);

    const Exponential<double> exponential = Exponential<double>(term);

    const Exponential<double> derivative = exponential.derivative(6);

    arma::vec rhs = arma::vec(10);

    for (arma::uword i = 0; i < 10; i++) {
      rhs(i) =
          wavenumbers(6) * arma::prod(arma::exp(grid.col(i) % wavenumbers));
    }

    CHECK(arma::approx_equal(at(derivative, grid), rhs, "abs_diff", 1e-10));

  }
}


TEST_CASE("Exponential Operators") {
  SECTION("One Dimension") {
    const arma::vec grid = arma::randu(10)/10;
    const arma::cx_vec cx_grid = arma::randu<arma::cx_vec>(10);

    const Exponential<double> exponential = Exponential<double>(arma::vec{1, 2},
                                                                arma::mat{{2, 4}});

    const auto derivative = exponential.derivative(0);

    for (arma::uword i = 0; i < grid.n_elem; i++) {
      CHECK(std::abs(
          (derivative * derivative).at(arma::vec{grid(i)})
            -
            std::pow(2. * std::exp(2 * grid(i)) + 8. * std::exp(4 * grid(i)),2))
            < 1e-10);
    }
  }

  SECTION("Multi Dimension") {
    const arma::mat grid = arma::randu<arma::mat>(10, 10);

    const arma::vec wavenumbers = arma::randu<arma::vec>(10);

    const auto term = exponential::Term<double>(1, wavenumbers);

    const Exponential<double> exponential = Exponential<double>(term);

    const Exponential<double> derivative = exponential.derivative(6);

    arma::vec rhs = arma::vec(10);

    for (arma::uword i = 0; i < 10; i++) {
      rhs(i) =
          wavenumbers(6) * arma::prod(arma::exp(grid.col(i) % wavenumbers));
    }

    CHECK(
        arma::approx_equal(at(derivative * derivative, grid), arma::pow(rhs, 2),
                           "abs_diff", 1e-10));

  }
}
}
