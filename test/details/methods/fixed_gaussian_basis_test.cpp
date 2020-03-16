#include <catch.hpp>

#include <quartz>

namespace quartz {
using namespace quartz::math;
using namespace quartz::method;

TEST_CASE("Fixed Gaussian Basis Test") {
  SECTION("One Dimension") {

    const Gaussian<double> real_space = Gaussian<double>(arma::mat{1.}, arma::vec{1.});
    const Polynomial<double> potential = Polynomial<double>(arma::vec{0.5},
                                                            lvec{2});

    const fgb::State test_state = fgb::State(real_space.wigner_transform(),
                                           arma::uvec{10,10},
                                           arma::mat{{-5., 5.},
                                                     {-5., 5.}},
                                           arma::vec{1.});

    const auto test_operator = fgb::Operator(test_state, potential);

    const Propagator<fgb::State> propagator =
        runge_kutta_4<fgb::Operator,
            fgb::State,
            Polynomial<double>>
            (test_operator, potential);


  }

}
}