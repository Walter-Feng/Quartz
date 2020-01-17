#define CATCH_CONFIG_MAIN

#include <Catch2/catch.hpp>

#include <quartz>

using namespace quartz::math;

TEST_CASE("Polynomial Value", "[factorial]") {
  SECTION("One Dimension") {
    const arma::vec grid = arma::linspace(-5, 5, 10);

    const Polynomial<double> squared = Polynomial<double>(arma::vec{0.5},
                                                          arma::umat{{{2}}});

    for (int i = 0; i < grid.n_elem; i++) {
      CHECK(squared.at(arma::vec{grid(i)}) == 0.5 * std::pow(grid(i), 2));
    }

    const Polynomial<double> cubed = Polynomial<double>(arma::vec{0.5},
                                                        arma::umat{{{3}}});

    for (int i = 0; i < grid.n_elem; i++) {
      CHECK(cubed.at(arma::vec{grid(i)}) == 0.5 * std::pow(grid(i), 3));
    }

    const Polynomial<double> squared_and_cubed = Polynomial<double>(
        arma::vec{0.5, 0.3},
        arma::umat{{{2, 3}}});

    for (int i = 0; i < grid.n_elem; i++) {
      CHECK(squared_and_cubed.at(arma::vec{grid(i)}) ==
            0.5 * std::pow(grid(i), 2) + 0.3 * std::pow(grid(i), 3));
    }
  }

  SECTION("Multi-Dimension") {
    const arma::mat grid = arma::randu<arma::mat>(10, 10);

    const arma::uvec indices = arma::randu<arma::uvec>(10);
    const Term<double> random_term = Term<double>{1,indices};

    const arma::vec result_lhs = random_term.at(grid);

    for (int i = 0; i < grid.n_cols; i++) {
      double result_rhs = 1;
      for(int j=0; j < grid.n_rows; j++) {

      }
      CHECK(result_lhs(i) == );
    }
  }
}
