#include <Catch2/catch.hpp>

#include <quartz>

namespace quartz {
using namespace quartz::math;
using namespace quartz::method;

TEST_CASE("MD Test") {
  SECTION("One Dimension") {

    const Gaussian real_space = Gaussian(arma::mat{1.}, arma::vec{1.});
    const Polynomial<double> potential = Polynomial<double>(arma::vec{0.5},
                                                            lvec{2});

    const md::State test_state = md::State(real_space.wigner_transform(),
                                           arma::uvec{20, 20},
                                           arma::mat{{-5., 5.},
                                                     {-5., 5.}},
                                           arma::vec{1.});

    const auto test_operator = md::Operator(test_state, potential);

    const Propagator<md::State> propagator =
        runge_kutta_4<md::Operator<Polynomial<double>>,
                      md::State,
                      Polynomial<double>>
            (test_operator, potential);


  }

}
}