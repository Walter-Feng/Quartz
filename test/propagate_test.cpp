#include <Catch2/catch.hpp>

#include <quartz>

namespace quartz {

TEST_CASE("Propagate") {

  std::cout << std::endl << std::endl;
  std::cout << "testing propagate function ..." << std::endl << std::endl;
  SECTION("Schrotinger") {

    const double dt = 0.01;

    const method::dvr::State initial_state =
        method::dvr::State(math::Gaussian(arma::mat{1.}, arma::vec{1}),
                           arma::uvec{100},
                           arma::mat{{-5, 5}});

    const auto harmonic_potential = math::Polynomial<double>(arma::vec{0.5},
                                                             lmat{2});

    const auto op = method::dvr::Operator(initial_state, harmonic_potential);

    const auto wrapper =
        math::schrotinger_wrapper<method::dvr::Operator,
                                  method::dvr::State,
                                  math::Polynomial<double>>;

    const auto result = propagate(initial_state,
                                  op,
                                  wrapper,
                                  harmonic_potential,
                                  generic_printer<method::dvr::State>, 10, dt,
                                  2);


  }

  SECTION("Classical Wigner") {

    const double dt = 0.01;

    const method::md::State initial_state =
        method::md::State(math::Gaussian(arma::mat{1.}, arma::vec{1}).wigner_transform(),
                          arma::uvec{30,30},
                          arma::mat{{-5, 5},{-5,5}});

    const auto harmonic_potential = math::Polynomial<double>(arma::vec{0.5},
                                                             lmat{2});

    const auto op = method::md::Operator(initial_state, harmonic_potential);

    const auto wrapper =
        math::runge_kutta_4<method::md::Operator<math::Polynomial<double>>,
            method::md::State,
            math::Polynomial<double>>;

    const auto result = propagate(initial_state,
                                  op,
                                  wrapper,
                                  harmonic_potential,
                                  generic_printer<method::md::State>, 10, dt,
                                  2);


  }
}

}
