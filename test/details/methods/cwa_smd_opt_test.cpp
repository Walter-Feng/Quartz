#include <catch.hpp>

#include <quartz>

namespace quartz {

TEST_CASE("Optimisation of cwa smd method") {

  using namespace method::cwa_smd_opt;

  const method::cwa_smd_opt::State initial_state =
      method::cwa_smd_opt::State(
          math::Gaussian<double>(arma::mat{1.}, arma::vec{
              2}).wigner_transform(),
          arma::uvec{10, 10},
          arma::mat{{-2, 2},
                    {-2, 2}}, 5);
  const arma::mat points = initial_state.points;

  const math::Polynomial<double> harmonic(arma::vec{1.0}, lmat{2});

  const auto h = hamiltonian(harmonic);

  const auto observable_list = polynomial_observables(2, 4);
  const arma::vec weights = initial_state.weights;
  const arma::vec scaling = initial_state.scaling;
  const arma::mat scaled_points = arma::diagmat(1.0 / scaling) * points;
  const arma::vec ref_expectation = initial_state.expectations;

  SECTION("Penalty Function") {
    const arma::mat perturbation = arma::randu(arma::size(points)) / 1e3;

    const arma::mat perturbed = points + perturbation;

    CHECK(
        std::pow(
            details::penalty_function(points, ref_expectation, observable_list,
                                      weights, scaling, 4), 2)
        < 1e-5);

  }

  SECTION("Penalty function derivative") {


    for (arma::uword i = 0; i < 100; i++) {

      const arma::vec rand_perturbation = (arma::vectorise(
          arma::randu(arma::size(points)) + 1));

      arma::vec perturbation = arma::vectorise(points) + rand_perturbation;
      const arma::mat ref_points = arma::reshape(perturbation,
                                                 arma::size(points));
      perturbation(i) += 1e-11;
      const arma::mat wrapped_perturbation = arma::reshape(perturbation,
                                                           arma::size(points));
      const arma::mat symbolic =
          details::penalty_function_derivative(ref_points, ref_expectation,
                                               observable_list, weights,
                                               scaling, 4);

      const auto numerical =
          (details::penalty_function(wrapped_perturbation, ref_expectation,
                                     observable_list, weights, scaling, 4)
           - details::penalty_function(ref_points, ref_expectation,
                                       observable_list, weights, scaling, 4)
          ) / 1e-11;

      if (std::abs(numerical) > 1e-5) {
        CHECK(std::abs(numerical - symbolic(i)) < 1e-4);
      }
    }

  }

  SECTION("a derivative minimization") {

    const arma::mat perturbation = arma::randu(arma::size(points)) + 1;
    details::cwa_smd_opt_param input;

    input.original_points = points + perturbation;
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
    const double gradient_tolerance = 0.01;
    const size_t total_steps = 1000;


    details::cwa_optimize(input, initial_step_size, tolerance,
                          gradient_tolerance, total_steps, "bfgs2");

  }

  SECTION("Initial state construction") {

    const auto harmonic_potential = math::Polynomial<double>(arma::vec{0.5},
                                                             lmat{2});

    const auto op = method::cwa_smd_opt::Operator(initial_state,
                                                  harmonic_potential);

    const auto wrapper =
        math::runge_kutta_4<method::cwa_smd_opt::Operator,
            method::cwa_smd_opt::State,
            math::Polynomial<double>>;

    const auto optimizer =
        method::cwa_smd_opt::cwa_opt<math::Polynomial<double>>(0.01, 0.1, 0.1,
                                                               100, "bfgs2");

    const auto all_wrapper = wrapper << optimizer;

    const auto printer = generic_printer<method::cwa_smd_opt::State>;

//    propagate(initial_state, op, all_wrapper, harmonic_potential, printer, 10,
//              0.01, 2);
  }
}

}