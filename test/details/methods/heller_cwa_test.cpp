#include <catch.hpp>

#include <quartz>

namespace quartz {

TEST_CASE("Heller's method in cwa case") {

  using namespace method::heller_cwa;

  SECTION("effective potential") {
    const math::Polynomial<cx_double> harmonic(arma::cx_vec{cx_double{1.0,0.0}}, lmat{2});
    CHECK(
        arma::approx_equal(method::heller_cwa::details::effective_potential(
            harmonic, cx_double{1.0}, cx_double{0.0}, cx_double{0.0}, 0,
            4).imag().exponents, lmat{1, 1}.t(),
                           "abs_diff", 1)
    );

  }

  SECTION("heller gaussian") {

    const arma::uvec grid = {5, 5};
    const arma::mat ranges = {{-1, 1},
                              {-1, 1}};
    const arma::mat points = math::space::points_generate(grid, ranges);

    const arma::cx_vec a = -arma::randu<arma::cx_vec>(6);
    const arma::cx_vec heller_gaussian_all = details::heller_gaussian(a,
                                                                      points);
    for (arma::uword i = 0; i < points.n_cols; i++) {
      const arma::vec point = points.col(i);
      const double p = point(1);
      const double q = point(0);
      CHECK(details::heller_gaussian(a, q, p) == heller_gaussian_all(i));
    }

    const auto gaussian = details::heller_gaussian(a);


    CHECK(arma::approx_equal(heller_gaussian_all, at(gaussian, points),
                             "abs_diff", 5e-16));


  }

  SECTION("E Function") {

    const math::Polynomial<cx_double> harmonic(arma::cx_vec{cx_double{1.0,0.0}}, lmat{2});

    const arma::cx_vec a = arma::randu<arma::cx_vec>(6);
    const arma::cx_vec a_derivatives = arma::randu<arma::cx_vec>(6);

    const auto V_eff_0 = method::heller_cwa::details::effective_potential(
        harmonic, cx_double{1.0}, cx_double{0.0}, cx_double{0.0}, 0, 4);

  }

  SECTION("I function derivative") {

    const arma::uvec grid = {5, 5};
    const arma::mat ranges = {{-1, 1},
                              {-1, 1}};
    const arma::mat points = math::space::points_generate(grid, ranges);

    const math::Polynomial<cx_double> harmonic(arma::cx_vec{cx_double{1.0,0.0}}, lmat{2});
    const auto V_eff_0 = method::heller_cwa::details::effective_potential(
        harmonic, 1.0, 0.0, 0.0, 0, 4);

    const arma::cx_vec a_derivatives_initial = -arma::randu<arma::cx_vec>(6);
    const arma::cx_vec a = arma::randu<arma::cx_vec>(6);

    for (arma::uword i = 0; i < 6; i++) {
      arma::cx_vec a_derivatives_perturb = a_derivatives_initial;
      a_derivatives_perturb(i) += 1e-9;
      const auto symbolic =
          details::I_function_derivative(a, a_derivatives_perturb, points,
                                         V_eff_0, 1);
      const auto numerical =
          -(details::I_function(a, a_derivatives_initial, points, V_eff_0, 1)
            - details::I_function(a, a_derivatives_perturb, points, V_eff_0, 1)
          ) / 1e-9;

      CHECK(std::abs(numerical - symbolic(i)) < 5e-4);
    }

    for (arma::uword i = 0; i < 6; i++) {
      arma::cx_vec a_derivatives_perturb = a_derivatives_initial;
      a_derivatives_perturb(i) += cx_double{0.00000, 1e-9};
      const auto symbolic =
          details::I_function_derivative(a, a_derivatives_perturb, points,
                                         V_eff_0, 1);
      const auto numerical =
          -(details::I_function(a, a_derivatives_initial, points, V_eff_0, 1)
            - details::I_function(a, a_derivatives_perturb, points, V_eff_0, 1)
          ) / 1e-9;

      CHECK(std::abs(numerical - symbolic(i + 6)) < 5e-4);
    }
  }

  SECTION("a derivative minimization") {

    const math::Polynomial<cx_double> harmonic(arma::cx_vec{cx_double{1.0,0.0}}, lmat{2});

    const arma::uvec grid = {5, 5};
    const arma::mat ranges = {{-1, 1},
                              {-1, 1}};
    const arma::mat points = math::space::points_generate(grid, ranges);

    details::HellerParam input;

    input.
        V_eff_0 = method::heller_cwa::details::effective_potential(
        harmonic, 1.0, 0.0, 0.0, 0, 4);
    input.
        points = points;
    input.
        a = arma::randu<arma::cx_vec>(6);
    input.
        mass = 1;

    const double initial_step_size = 0.01;
    const double tolerance = 0.1;
    const double gradient_tolerance = 0.1;
    const size_t total_steps = 100;


    details::a_derivative(input, initial_step_size, tolerance,
                          gradient_tolerance, total_steps);

  }
}

}