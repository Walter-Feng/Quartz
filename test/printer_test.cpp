#include <catch.hpp>

#include <quartz>

namespace quartz {

TEST_CASE("Check printer") {

  std::cout << "Test printer ... " << std::endl << std::endl;

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

    printer(test, 0, 0.0, 1, true);
    for (int i = 0; i < 5; i++) {
      printer(test, i + 1, 0.1 * (i + 1), 1, false);
    }

    std::cout << std::endl;

    printer(test, 0, 0, 2, true);
    for (int i = 0; i < 5; i++) {
      printer(test, i + 1, 0.1 * (i + 1), 2, false);
    }

    std::cout << std::endl;

    printer(test, 0, 0, 3, true);
    for (int i = 0; i < 5; i++) {
      printer(test, i + 1, 0.1 * (i + 1), 3, false);
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

    printer(test, 0, 0, 1, true);
    for (int i = 0; i < 5; i++) {
      printer(test, i + 1, 0.1 * (i + 1), 1, false);
    }
  }

  SECTION("Expectation printer") {
    struct dummy_state {
      [[nodiscard]] arma::vec expectation(
          const std::vector<math::Polynomial<double>> & function) const {

        return arma::randu<arma::vec>(function.size());
      }
    } test;

    const auto operators = polynomial_observables(1, 5);

    const auto printer = expectation_printer<dummy_state>(operators);

    printer(test, 0, 0, 1, true);
    for (int i = 0; i < 5; i++) {
      printer(test, i + 1, 0.1 * (i + 1), 1, false);
    }
  }
}

}