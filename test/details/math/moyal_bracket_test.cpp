#include <catch.hpp>

#include <quartz>

namespace quartz {
namespace math {

TEST_CASE("Moyal Bracket") {

  SECTION("One Dimension") {

    const auto x = Polynomial(arma::vec{1}, lvec{1,0});
    const auto p = Polynomial(arma::vec{1}, lvec{0,1});
    const auto x2 = x.pow(2);
    const auto p2 = p.pow(2);
    const auto xp = x*p;

    const auto kinetic = kinetic_energy<double>(1);

    const auto moyal = moyal_bracket(x, kinetic, 3);

    CHECK(moyal.coefs(0) == 1);
    CHECK(moyal.indices(1,0) == 1);

    const auto harmonic_potential = Polynomial<double>(arma::vec{0.5}, lvec{2});
    const auto h2 = hamiltonian(harmonic_potential);

    CHECK(moyal_bracket(p, h2, 3).coefs(0) == -1);
    CHECK(moyal_bracket(p, h2, 3).indices(0,0) == 1);

  }
}

}
}