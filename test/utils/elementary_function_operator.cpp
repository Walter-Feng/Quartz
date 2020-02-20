#include <Catch2/catch.hpp>

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
}

}
