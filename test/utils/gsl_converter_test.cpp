#include <catch.hpp>

#include <quartz>

namespace quartz {

TEST_CASE("GSL type converter") {

  SECTION("GSL vector") {
    const arma::vec test = arma::randu(5);

    std::cout << test.t();

    const auto gsl_test = *gsl::convert_vec(test);
    for(arma::uword i=0; i<gsl_test.size; i++) {
      std::cout << gsl_vector_get(&gsl_test, i) << "   ";
    }

    std::cout << std::endl;
  }

}


}