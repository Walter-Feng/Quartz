#include <catch.hpp>

#include <quartz>

namespace quartz {

TEST_CASE("Heller's method in cwa case") {

  using namespace method::cwa_smd_opt;

  const arma::uvec grid = {5, 5};
  const arma::mat ranges = {{-1, 1},
                            {-1, 1}};
  const arma::mat points = math::space::points_generate(grid, ranges);

  const math::Polynomial<double> harmonic(arma::vec{1.0}, lmat{2});

  const auto h = hamiltonian(harmonic);

  const
  SECTION("Penalty Function") {

  }

  SECTION("I function derivative") {

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