#include <Catch2/catch.hpp>

#include <quartz>

namespace quartz {

TEST_CASE("Propagate") {
  SECTION("Schrotinger") {

    const double dt = 0.01;

    const method::dvr::State initial_state =
        method::dvr::State(math::Gaussian(arma::mat{1.}, arma::vec{1}),
                           arma::uvec{10},
                           arma::mat{{-5, 5}});

    const auto harmonic_potential = math::Polynomial<double>(arma::vec{0.5},
                                                             lmat{1});

    const auto op = method::dvr::Operator(initial_state, harmonic_potential);

    const auto wrapper =
        math::schrotinger_wrapper<method::dvr::Operator,
                                  method::dvr::State,
                                  math::Polynomial<double>>;

    const auto result = propagate(initial_state,
              op,
              wrapper,
              harmonic_potential,
              generic_printer<method::dvr::State>, 3, dt);


  }
}

}
