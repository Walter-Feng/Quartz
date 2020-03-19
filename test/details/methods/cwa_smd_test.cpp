#include <catch.hpp>

#include <quartz>

namespace quartz {

TEST_CASE("Classical Wigner Approximation + Semi Moyal Dynamics") {

  const method::cwa_smd::State initial_state =
      method::cwa_smd::State(math::Gaussian<double>(arma::mat{1.}, arma::vec{
                                 1}).wigner_transform(),
                             arma::uvec{30, 30},
                             arma::mat{{-5, 5},
                                       {-5, 5}}, 3);

  const auto harmonic_potential = math::Polynomial<double>(arma::vec{0.5},
                                                           lmat{2});

  const auto op = method::cwa_smd::Operator(initial_state, harmonic_potential);

  const auto wrapper =
      math::runge_kutta_4<method::cwa_smd::Operator,
          method::cwa_smd::State,
          math::Polynomial<double>>;

}

}