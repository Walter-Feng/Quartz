#include <Catch2/catch.hpp>

#include <quartz>

namespace quartz {
using namespace quartz::math;
using namespace quartz::method;

TEST_CASE("WD Test") {
  SECTION("One Dimension") {

    const Gaussian<double> real_space = Gaussian<double>(arma::mat{1.},
                                                         arma::vec{1.});
    const Polynomial<double> potential = Polynomial<double>(arma::vec{0.5},
                                                            lvec{2});

    const md::State test_state = wd::State(
        GaussianWithPoly(real_space.wigner_transform()),
        arma::uvec{20, 20},
        arma::mat{{-5., 5.},
                  {-5., 5.}},
        arma::vec{1.});

    const auto test_operator = wd::Operator(test_state,
                                            GaussianWithPoly(
                                                real_space.wigner_transform()),
                                            potential);

    const Propagator<wd::State> propagator =
        runge_kutta_4<wd::Operator<Polynomial<double>, GaussianWithPoly<double>>,
            wd::State,
            Polynomial<double>>
            (test_operator, potential);


  }

}
}