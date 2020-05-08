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

  const arma::vec scaling = arma::ones(2);

  SECTION("Arma basic tests") {
    const arma::mat original = arma::randu(2,50);
    CHECK(arma::approx_equal(original, arma::reshape(arma::vectorise(original),2,50), "abs_diff", 1e-16));
  }

  SECTION("Penalty Function") {
    const arma::mat perturbation = arma::randu(2, 25) / 1e3;

    const arma::mat perturbed = points + perturbation;

    CHECK(
        details::penalty_function(points, ref_expectation, observable_list, weights, scaling, 4)
         == 0);

  }

  SECTION("Penalty function derivative") {

    for (arma::uword i = 0; i < 50; i++) {


      const arma::vec rand_perturbation = (arma::randu(50) + 1) / 100;

      arma::vec perturbation = arma::vectorise(points) + rand_perturbation;
      const arma::mat ref_points = arma::reshape(perturbation, 2, 25);
      perturbation(i) += 1e-7;
      const arma::mat wrapped_perturbation = arma::reshape(perturbation, 2, 25);
      const arma::mat symbolic =
          details::penalty_function_derivative(ref_points,ref_expectation, observable_list, weights, scaling, 4);

//      const auto numerical =
//          (details::penalty_function(wrapped_perturbation, ref_expectation, observable_list, weights, scaling, 4)
//            - details::penalty_function(ref_points, ref_expectation, observable_list, weights, scaling, 4)
//          ) / 1e-7;

//      std::cout << symbolic(i) << " v.s. " << numerical  << std::endl;
//      CHECK(std::abs(numerical - symbolic(i)) / std::abs(symbolic(i)) < 5e-4);
    }

    }
//
  SECTION("a derivative minimization") {


    details::cwa_smd_opt_param input;

    input.original_points = points;
    input.
        expectations_ref = ref_expectation;
    input.
        original_operators = observable_list;
    input.
        weights = weights;
    input.
        scaling = scaling;
    input.grade = 4;

    const double initial_step_size = 0.01;
    const double tolerance = 0.1;
    const double gradient_tolerance = 0.1;
    const size_t total_steps = 100;


    details::cwa_optimize(input, initial_step_size, tolerance,
                          gradient_tolerance, total_steps).print();

  }
}

}