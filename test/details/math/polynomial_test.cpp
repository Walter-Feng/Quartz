#include <catch.hpp>

#include <quartz>

namespace quartz {

using namespace quartz::math;

TEST_CASE("Polynomial Value") {
  SECTION("One Dimension") {
    const arma::vec grid = arma::linspace(-5, 5, 10);
    const arma::cx_vec cx_grid = arma::cx_vec{arma::linspace(-5, 5, 10),
                                              arma::zeros(10)};

    const Polynomial<double> squared = Polynomial<double>(arma::vec{0.5},
                                                          lmat{{{2}}});

    for (int i = 0; i < grid.n_elem; i++) {
      CHECK(squared.at(arma::vec{grid(i)}) == 0.5 * std::pow(grid(i), 2));
    }

    for (int i = 0; i < grid.n_elem; i++) {
      CHECK(
          std::abs(std::real(
              squared.at(
                  arma::cx_vec{cx_grid(i)})) -
          0.5 * std::pow(grid(i), 2)) < 1e-14);
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
    const polynomial::Term<double>
        random_term = polynomial::Term<double>{1, indices};

    arma::vec result_lhs = arma::vec(10);
    arma::vec result_rhs = arma::vec(10);
    for (int i = 0; i < 10; i++) {
      const arma::vec each_position = grid.col(i);
      result_lhs(i) = random_term.at(each_position);
      double result_rhs_temp = 1;
      for (int j = 0; j < 10; j++) {
        result_rhs_temp *= std::pow(grid(j), indices(j));
      }
      result_rhs(i) = result_rhs_temp;
    }

    CHECK(arma::approx_equal(result_lhs, result_rhs, "abs_diff", 1e-14));

  }
}

TEST_CASE("Operators") {
  const arma::vec grid = arma::linspace(-5, 5, 10);
  const arma::cx_vec cx_grid = arma::cx_vec{arma::linspace(-5, 5, 10),
                                            arma::zeros(10)};

  const Polynomial<double> squared = Polynomial<double>(arma::vec{0.5},
                                                        lmat{{{2}}});

  const Polynomial<cx_double>
      cx_cubed = Polynomial<cx_double>(arma::cx_vec{0.3}, lmat{{{3}}});

  SECTION("Summation") {

    const Polynomial<double> squared_and_cubed = Polynomial<double>(
        arma::vec{0.5, 0.3},
        lmat{{{2, 3}}});

    const auto sum = squared + cx_cubed;

    for (int i = 0; i < grid.n_elem; i++) {
      CHECK(sum.at(arma::vec{grid(i)}) ==
            squared_and_cubed.at(arma::vec{grid(i)}));
    }

    for (int i = 0; i < grid.n_elem; i++) {
      CHECK(sum.at(arma::cx_vec{cx_grid(i)}) ==
            squared_and_cubed.at(arma::cx_vec{cx_grid(i)}));
    }
  }

  SECTION("Multiplication") {
    const auto product = squared * cx_cubed;

    const auto squared_by_cubed =
        Polynomial<cx_double>(arma::cx_vec{0.15}, lmat{{{5}}});

    for(int i=0; i<grid.n_elem; i++) {
      CHECK(product.at(arma::vec{grid(i)}) == squared_by_cubed.at(arma::vec{grid(i)}));
    }
  }

  SECTION("Subtraction") {
    const auto subtracted = squared - cx_cubed;
    const auto squared_without_cubed = Polynomial<double>(
        arma::vec{0.5, -0.3},
        lmat{{{2, 3}}});

    for (int i = 0; i < grid.n_elem; i++) {
      CHECK(subtracted.at(arma::vec{grid(i)}) ==
            squared_without_cubed.at(arma::vec{grid(i)}));
    }

    for (int i = 0; i < grid.n_elem; i++) {
      CHECK(subtracted.at(arma::cx_vec{cx_grid(i)}) ==
            squared_without_cubed.at(arma::cx_vec{cx_grid(i)}));
    }
  }

  SECTION("Combination") {
    const auto sum = squared + cx_cubed;
    const auto squared_without_cubed = Polynomial<double>(
        arma::vec{0.5, -0.3},
        lmat{{{2, 3}}});
    const auto product = sum * squared_without_cubed;

    const auto result = Polynomial<cx_double>(
        arma::cx_vec{0.25, -0.09},
        lmat{{{4, 6}}});

    for (int i = 0; i < grid.n_elem; i++) {
      CHECK(std::abs(product.at(arma::vec{grid(i)}) -
            result.at(arma::vec{grid(i)})) < 1e-14);
    }

    for (int i = 0; i < grid.n_elem; i++) {
      CHECK(std::abs(product.at(arma::cx_vec{cx_grid(i)}) -
            result.at(arma::cx_vec{cx_grid(i)})) < 1e-14);
    }
  }
}

TEST_CASE("Member Functions") {
  SECTION("Grades") {
    const Polynomial<double> squared = Polynomial<double>(arma::vec{0.5},
                                                          lmat{{{2}}});

    CHECK(squared.grade() == 2);
  }
}


}