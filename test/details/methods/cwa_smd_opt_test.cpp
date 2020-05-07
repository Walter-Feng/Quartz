#include <catch.hpp>

#include <quartz>

namespace quartz {

TEST_CASE("Optimisation of cwa smd method") {

  using namespace method::cwa_smd_opt;

  const arma::uvec grid = {5, 5};
  const arma::mat ranges = {{-1, 1},
                            {-1, 1}};
  const arma::mat points = math::space::points_generate(grid, ranges);

  const math::Polynomial<double> harmonic(arma::vec{1.0}, lmat{2});

  const auto h = hamiltonian(harmonic);

  const auto observable_list = polynomial_observables(2,4);
  const arma::vec weights = arma::ones<arma::vec>(points.n_cols);
  const method::cwa::State ref_state(points, arma::ones<arma::vec>(points.n_cols), arma::vec{1});
  const arma::vec ref_expectation = ref_state.expectation(observable_list);

  SECTION("Penalty Function") {

    const arma::vec scaling = arma::ones(2);
    const arma::mat perturbation = arma::randu(2, 25) / 1e3;

    const arma::mat perturbed = points + perturbation;

    CHECK(
        details::penalty_function(points, ref_expectation, observable_list, weights, scaling, 4)
         == 0);

  }

  SECTION("I function derivative") {

    }
//
//  SECTION("a derivative minimization") {
//
//    const math::Polynomial<cx_double> harmonic(arma::cx_vec{cx_double{1.0,0.0}}, lmat{2});
//
//    const arma::uvec grid = {5, 5};
//    const arma::mat ranges = {{-1, 1},
//                              {-1, 1}};
//    const arma::mat points = math::space::points_generate(grid, ranges);
//
//    details::HellerParam input;
//
//    input.
//        V_eff_0 = method::heller_cwa::details::effective_potential(
//        harmonic, 1.0, 0.0, 0.0, 0, 4);
//    input.
//        points = points;
//    input.
//        a = arma::randu<arma::cx_vec>(6);
//    input.
//        mass = 1;
//
//    const double initial_step_size = 0.01;
//    const double tolerance = 0.1;
//    const double gradient_tolerance = 0.1;
//    const size_t total_steps = 100;
//
//
//    details::a_derivative(input, initial_step_size, tolerance,
//                          gradient_tolerance, total_steps);
//
//  }
}

}