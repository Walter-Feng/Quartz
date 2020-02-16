#include <Catch2/catch.hpp>

#include <quartz>

namespace quartz {

TEST_CASE("Check printer") {

  SECTION("One Dimension") {
    struct dummy_state {
      arma::vec positional_expectation() const {
        return arma::randu<arma::vec>(1);
      }

      arma::vec momentum_expectation() const {
        return -arma::randu<arma::vec>(1);
      }
    } test;

    const auto printer = generic_printer<dummy_state>;

    printer(test, 0.0, 1, true);
    for (int i = 0; i < 5; i++) {
      printer(test, 0.1 * (i + 1), 1, false);
    }

    std::cout << std::endl;

    printer(test, 0, 2, true);
    for (int i = 0; i < 5; i++) {
      printer(test, 0.1 * (i + 1), 2, false);
    }

    std::cout << std::endl;

    printer(test, 0, 3, true);
    for (int i = 0; i < 5; i++) {
      printer(test, 0.1 * (i + 1), 3, false);
    }
    std::cout << std::endl;
  }

  SECTION("Generic printer") {
    struct dummy_state {
      arma::vec positional_expectation() const {
        return arma::randu<arma::vec>(5);
      }

      arma::vec momentum_expectation() const {
        return -arma::randu<arma::vec>(5);
      }
    } test;

    const auto printer = generic_printer<dummy_state>;

    printer(test, 0, 1, true);
    for (int i = 0; i < 5; i++) {
      printer(test, 0.1 * (i + 1), 1, false);
    }
  }

}

}