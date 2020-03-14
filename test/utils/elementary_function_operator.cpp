#include <catch.hpp>

#include <quartz>

namespace quartz {

TEST_CASE("Elementary Function Operator") {
  SECTION("Exponential") {
    const double arg = arma::randu();

    const auto multiply = [arg](const double B) {
      return arg * B;
    };

    CHECK(std::abs(exp(multiply, 1.0, 100) - std::exp(arg)) < 1e-10);
  }

  SECTION("Sinusoidal") {
    const double arg = arma::randu();

    const auto multiply = [arg](const double B) {
      return arg * B;
    };

    CHECK(std::abs(sin(multiply, 1.0, 500) - std::sin(arg)) < 1e-10);
    CHECK(std::abs(cos(multiply, 1.0, 500) - std::cos(arg)) < 1e-10);
  }
}

}
