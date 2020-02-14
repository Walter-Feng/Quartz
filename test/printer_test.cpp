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

    generic_printer(test, 0.0, 1, true);
    for (int i = 0; i < 5; i++) {
      generic_printer(test, 0.1 * (i + 1), 1);
    }

    std::cout << std::endl;

    generic_printer(test, 0, 2, true);
    for (int i = 0; i < 5; i++) {
      generic_printer(test, 0.1 * (i + 1), 2);
    }

    std::cout << std::endl;

    generic_printer(test, 0, 3, true);
    for (int i = 0; i < 5; i++) {
      generic_printer(test, 0.1 * (i + 1), 3);
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

    generic_printer(test, 0, 1, true);
    for (int i = 0; i < 5; i++) {
      generic_printer(test, 0.1 * (i + 1), 1);
    }
  }

}

}