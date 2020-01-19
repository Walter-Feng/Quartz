#define CATCH_CONFIG_MAIN

#include <Catch2/catch.hpp>

#include <quartz>

namespace quartz {

using namespace quartz::math;

TEST_CASE("Polynomial Value", "[factorial]") {
  SECTION("One Dimension") {
    const arma::vec grid = arma::linspace(-5, 5, 10);

    const Polynomial<double> squared = Polynomial<double>(arma::vec{0.5},
                                                          lmat{{{2}}});

    for (int i = 0; i < grid.n_elem; i++) {
      CHECK(squared.at(arma::vec{grid(i)}) == 0.5 * std::pow(grid(i), 2));
    }

    const Polynomial<double> cubed = Polynomial<double>(arma::vec{0.5},
                                                        lmat{{{3}}});

    for (int i = 0; i < grid.n_elem; i++) {
      CHECK(cubed.at(arma::vec{grid(i)}) == 0.5 * std::pow(grid(i), 3));
    }

    const Polynomial<double> squared_and_cubed = Polynomial<double>(
        arma::vec{0.5, 0.3},
        lmat{{{2, 3}}});

    for (int i = 0; i < grid.n_elem; i++) {
      CHECK(squared_and_cubed.at(arma::vec{grid(i)}) ==
            0.5 * std::pow(grid(i), 2) + 0.3 * std::pow(grid(i), 3));
    }
  }

  SECTION("Complex") {
    const arma::vec grid = arma::linspace(-5, 5, 10);

    const Polynomial<arma::cx_double> squared =
        Polynomial<arma::cx_double>(
            arma::cx_vec{0.5},
            lmat{{{2}}}
        );

    for (int i = 0; i < grid.n_elem; i++) {
      CHECK(std::real(squared.at(arma::vec{grid(i)})) ==
            0.5 * std::pow(grid(i), 2));
    }

    const Polynomial<arma::cx_double> cubed =
        Polynomial<arma::cx_double>(
            arma::cx_vec{0.5},
            lmat{{{3}}}
        );

    for (int i = 0; i < grid.n_elem; i++) {
      CHECK(std::real(cubed.at(arma::vec{grid(i)})) ==
            0.5 * std::pow(grid(i), 3));
    }

    const Polynomial<arma::cx_double> squared_and_cubed =
        Polynomial<arma::cx_double>(
            arma::cx_vec{0.5, 0.3},
            lmat{{{2, 3}}}
        );

    for (int i = 0; i < grid.n_elem; i++) {
      CHECK(std::real(squared_and_cubed.at(arma::vec{grid(i)})) ==
            0.5 * std::pow(grid(i), 2) + 0.3 * std::pow(grid(i), 3));
    }
  }

  SECTION("Multi-Dimension") {
    const arma::mat grid = arma::randu<arma::mat>(10, 10);

    const lvec indices = lvec{3, 1, 5, 12, 5, 21, 2, 1, 7, 34};
    const polynomial::Term<double> random_term = polynomial::Term<double>{1,indices};

    arma::vec result_lhs = arma::vec(10);
    arma::vec result_rhs = arma::vec(10);
    for (int i = 0; i < 10; i++) {
      result_lhs(i) = random_term.at(grid.col(i));
      double result_rhs_temp = 1;
      for (int j = 0; j < 10; j++) {
        result_rhs_temp *= std::pow(grid(j), indices(j));
      }
      result_rhs(i) = result_rhs_temp;
    }

    CHECK(arma::approx_equal(result_lhs, result_rhs, "abs_diff", 1e-16));

  }
}

}