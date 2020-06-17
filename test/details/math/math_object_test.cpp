#include <catch.hpp>

#include <quartz>

namespace quartz {

TEST_CASE("Elementary Function") {

  ElementaryFunction<double>
      test = {math::Polynomial<double>(arma::vec{1.0}, lmat{2})};

  CHECK(test.at(arma::vec{2.0}) == 4.0);
  CHECK(test.derivative(0).at(arma::vec{3.0}) == 6.0);

  ElementaryFunction<double>
      exponential = {math::Exponential(arma::vec{1.0}, arma::mat{2.0})};

  CHECK(exponential.at(arma::vec{1.0}) == std::exp(2.0));
  CHECK(exponential.derivative(0).at(arma::vec{1.0}) == 2.0 * std::exp(2.0));

  ElementaryFunction<double> number = {1.0};
  CHECK(number.at(arma::vec{1.0}) == 1.0);
  CHECK(number.derivative(0).at(arma::vec{1.0}) == 0.0);
}

TEST_CASE("MathObject") {

  const arma::vec test_point = arma::vec{3.0};

  ElementaryFunction<double>
      test = {math::Polynomial<double>(arma::vec{1.0}, lmat{2})};
  ElementaryFunction<double>
      test2 = {math::Polynomial<double>(arma::vec{2.0}, lmat{3})};

  const auto a = MathObject<double>(test);

  CHECK(a.at(test_point) == 9.0);
  CHECK(a.derivative(0).at(test_point) == 6.0);

  const MathObject<double> test_left = test;
  const MathObject<double> test_right = test2;

  MathObject<double> test_unique(test_left, test_right, math::OperatorType::Sum);

  CHECK(test_unique.at(test_point) == 63.0);

  MathObject<double> cloned_test = test_unique;
  CHECK(cloned_test.at(test_point) == 63.0);
  CHECK(cloned_test.derivative(0).at(test_point) == 6.0 + 54.0);

  cloned_test.type = math::OperatorType::Subtract;
  CHECK(cloned_test.at(test_point) == 9.0 - 54.0);
  CHECK(cloned_test.derivative(0).at(test_point) == 6.0 - 54.0);

  cloned_test.type = math::OperatorType::Multiply;
  CHECK(cloned_test.at(test_point) == 9.0 * 54.0);
  CHECK(cloned_test.derivative(0).at(test_point) == 9.0 * 54.0 + 6.0 * 54.0);

  MathObject<double> reclone = cloned_test;
  CHECK(reclone.at(test_point) == 9.0 * 54.0);
  CHECK(cloned_test.at(test_point) == 9.0 * 54.0);

  MathObject<double> cloned_test_plus_reclone = MathObject(cloned_test, reclone,
                                             math::OperatorType::Sum);
  CHECK(cloned_test_plus_reclone.at(test_point) == 2.0 * 9.0 * 54.0);
  CHECK(cloned_test_plus_reclone.at(test_point) ==
        cloned_test.at(test_point)
        + reclone.at(test_point));
  CHECK((cloned_test + reclone).at(test_point) == cloned_test.at(test_point)
                                                      + reclone.at(test_point));
  CHECK((cloned_test * reclone).at(test_point) == cloned_test.at(test_point)
                                                  * reclone.at(test_point));
}
}