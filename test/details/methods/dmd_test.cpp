#include <catch.hpp>

#include <quartz>

namespace quartz {
using namespace quartz::math;
using namespace quartz::method;

TEST_CASE("DMD Test") {
  SECTION("One Dimension") {

    const Gaussian<double> real_space = Gaussian<double>(arma::mat{1.},
                                                         arma::vec{1.});
    const Polynomial<double> potential = Polynomial<double>(arma::vec{0.5},
                                                            lvec{2});

    const cwa::State test_state = dmd::State(
        GaussianWithPoly(real_space.wigner_transform()),
        arma::uvec{20, 20},
        arma::mat{{-5., 5.},
                  {-5., 5.}},
        arma::vec{1.});

    const auto test_operator = dmd::Operator(test_state,
                                            potential);

    const Propagator<dmd::State> propagator =
        runge_kutta_4<dmd::Operator<Polynomial<double>>,
            dmd::State,
            Polynomial<double>>
            (test_operator, potential);


  }

}
}