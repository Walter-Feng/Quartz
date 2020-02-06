#include <Catch2/catch.hpp>

#include <quartz>

namespace quartz {

using namespace quartz::math;

TEST_CASE("Check members") {

  SECTION("Check has_time_evolve") {
    struct td_potential {

    private:
      double time;

    public:
      void time_evolve(const double & dt) {
        this->time += dt;
      }
    };

    struct constant_potential {

    };

    CHECK(has_time_evolve<td_potential, void(const double &)>::value);

    CHECK(!has_time_evolve<constant_potential, void(const double &)>::value);
  }

  SECTION("Check has_derivative") {

    struct dummy_function {

    };
    CHECK(!has_derivative<dummy_function, void(const double &)>::value);

    CHECK(has_derivative<Polynomial<cx_double>,
              Polynomial<cx_double>(const arma::uword &)>::value);

    CHECK(!has_derivative<Polynomial<double>,
              Polynomial<cx_double>(const arma::uword &)>::value);

    CHECK(has_derivative<Sinusoidal<double>,
              Sinusoidal<double>(const arma::uword &)>::value);
  }


  SECTION("Check has_at") {
    struct dummy_function {

    };

    CHECK(!has_at<dummy_function, double(const arma::vec &)>::value);

    CHECK(has_at<Polynomial<cx_double>,
              cx_double(const arma::cx_vec &)>::value);

    CHECK(has_at<Polynomial<double>,
        cx_double(const arma::cx_vec &)>::value);

    CHECK(has_at<Sinusoidal<double>,
              double(const arma::vec &)>::value);
  }

}

}

