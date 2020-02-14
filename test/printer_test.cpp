#include <Catch2/catch.hpp>

#include <quartz>

namespace quartz {

TEST_CASE("Check printer") {
  SECTION("Generic printer") {
    struct dummy_state {
      arma::vec positional_expectation() const {
        return arma::randu<arma::vec>(5);
      }

      arma::vec momentum_expectation() const {
        return -arma::randu<arma::vec>(5);
      }
    } test;

    generic_printer(test,1, true);
    for(int i=0; i<5; i++) {
      generic_printer(test,1);
    }

    std::cout << std::endl;

    generic_printer(test,2, true);
    for(int i=0; i<5; i++) {
      generic_printer(test,2);
    }

    std::cout << std::endl;

    generic_printer(test,3, true);
    for(int i=0; i<5; i++) {
      generic_printer(test,3);
    }
    std::cout << std::endl;
  }
}

}