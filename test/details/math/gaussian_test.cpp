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

    const Gaussian<double> gaussian2 = Gaussian<double>(
        arma::eye<arma::mat>(1, 1));

    for (arma::uword i = 0; i < grid.n_elem; i++) {
      CHECK(std::real(gaussian2.at(arma::vec{grid(i)})) ==
            std::exp(-0.5 * grid(i) * grid(i)));
    }

    const Gaussian<double> gaussian3 = Gaussian(arma::eye<arma::mat>(1, 1),
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

    const Gaussian<double> gaussian2 = Gaussian<double>(arma::eye<arma::mat>(1, 1));

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

    const arma::mat covariance = arma::mat{{1,   0.1},
                                           {0.1, 1}};
    const arma::vec mean = arma::vec{1, 2};
    const Gaussian<double> gaussian_from_mean_and_covariance =
        Gaussian<double>(arma::inv(covariance), arma::inv(covariance) * mean);

    CHECK(arma::approx_equal(covariance,
                             gaussian_from_mean_and_covariance.covariance(),
                             "abs_diff", 1e-16));
    CHECK(arma::approx_equal(mean, arma::real(
        gaussian_from_mean_and_covariance.mean()), "abs_diff", 1e-16));
  }
}

TEST_CASE("Gaussian Integral") {
  SECTION("Real space") {
    const Gaussian<double> test = Gaussian<double>(arma::mat{1});
    const Gaussian<double> wigner_transform = test.wigner_transform();

    CHECK(test.integral() == std::sqrt(2. * pi));
    CHECK(
        std::abs((test * test).integral() - wigner_transform.integral()) < 1e-14);

    const Polynomial<double> polynomial_const = Polynomial<double>(1, 1.0);
    CHECK(test.integral(polynomial_const) == test.integral());

    const Polynomial<double> polynomial_null = Polynomial<double>(1);
    CHECK(test.integral(polynomial_null) == 0.);

    const Polynomial<double> polynomial_1 = Polynomial<double>(arma::vec{1.0},
                                                               lvec{1});
    CHECK(test.integral(polynomial_1) == 0.);

    const Polynomial<double> polynomial_1_with_const =
        Polynomial<double>(arma::vec{1.0, 1.0}, lmat{{1, 0}});
    CHECK(test.integral(polynomial_1_with_const) == test.integral());

    const Polynomial<double> polynomial_4 =
        Polynomial<double>(arma::vec{1.0, 4.0}, lmat{{1, 0}});

    CHECK(
        std::abs(test.integral(polynomial_4) - cx_double{10.026513098524, 0.0}) <
        1e-14);

    const Gaussian<double> test_2 = Gaussian<double>(arma::mat{2});
    const Gaussian<double> wigner_transform_2 = test_2.wigner_transform();

    CHECK(test_2.integral() == std::sqrt(pi));
    CHECK(std::abs((test_2 * test_2).integral() - wigner_transform_2.integral()) <
          1e-14);

    const Gaussian<double> with_monomial = Gaussian<double>(arma::mat{1}, arma::vec{2});
    const Gaussian<double> wigner_transform_monomial = with_monomial.wigner_transform();
    CHECK(with_monomial.integral() == std::sqrt(2. * pi) * std::exp(2));
    CHECK((with_monomial * with_monomial).integral() ==
          wigner_transform_monomial.integral());


    const Gaussian<double> test_multi_1 = Gaussian<double>(arma::mat{{3, 0},
                                                                     {0, 2}}, arma::vec{1, 2});
    const Gaussian<double> wigner_transform_multi_1 = test_multi_1.wigner_transform();
    CHECK(std::abs(test_multi_1.integral() -
                   std::sqrt(2.0 / 3.0) * std::exp(7.0 / 6.0) * pi) < 1e-14);
    CHECK(std::abs((test_multi_1 * test_multi_1).integral()
                   - wigner_transform_multi_1.integral()) < 1e-14);

    const Gaussian<double> test_multi = Gaussian<double>(arma::mat{{6, 1},
                                                                   {1, 4}}, arma::vec{1, 2});
    const Gaussian<double> wigner_transform_multi = test_multi.wigner_transform();
    CHECK((test_multi * test_multi).integral() ==
          wigner_transform_multi.integral());
  }

  SECTION("With phase factors") {

    const auto test_complex = Gaussian(arma::mat{1.},
                                       arma::cx_vec{arma::vec{1},arma::vec{2}});

    const auto WT_test = test_complex.wigner_transform();

    const auto mean = WT_test.mean();
    CHECK(std::real(mean(1)) == 2);

  }
}


}
