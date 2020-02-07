// name your method
#ifndef METHOD_METHOD_TEMPLATE_H
#define METHOD_METHOD_TEMPLATE_H

// include only the necessary header files
#include "propagate.h"
#include "details/math/polynomial.h"

namespace method {
// use your method name to create a subspace for your
// implementation of details
namespace method_template {

namespace details {
// use constant expression(const) and reference(&)
// as much as possible
// if you are not implementing a function with templates,
// type inline
inline
double detail_A(const ExampleState & test_state) {
  return 1.0;
}

// directly type the namespace for function calling
// instead of "using namespace ..."
// as much as possible
template<typename T>
math::Polynomial<T> detail_B(const arma::Col<T> & a_vector,
                             const double some_parameter) {
  return math::Polynomial<T>(1);
}
} // namespace details

template<typename T>
Propagator<ExampleState<T>, math::Polynomial<T>, T>
    propagator(const State & state,
               const Polynomial<T> & potential,
               const double dt) {
  // State your requirement of potential, for example,
  // you will only need the potential defined over the real space (such as DVR),
  // by requiring the potential (Polynomial<T> type here)
  // to have .at() as a member function,
  // or you might need the second derivative,
  // by requiring the potential
  // to have .derivative(arma::Col<T>) as a member function.

  // your implementation here


}

} // namespace method_template

// It is suggested to satisfy both complex (cx_double) and real (double)
// and suitable for arbitrary dimensions
template<typename T>
struct ExampleState {
public:
  arma::Col <T> vector_as_an_example;
  arma::Mat <T> matrix_as_an_example_but_not_recommended;
  double a_parameter;
  double b_parameter;
  bool is_this_struct_satisfy_something;

  // Establish an easy way to construct your State
  inline
  ExampleState(const arma::Col <T> vector_for_state_construction,
               const arma::Mat <T> matrix_for_state_construction,
               const double a_parameter,
               const double b_parameter,
               const bool turn_on_something = true) :

      vector_as_an_example(vector_for_state_construction),
      matrix_as_an_example_but_not_recommended(matrix_for_state_construction),
      a_parameter(a_parameter),
      b_parameter(b_parameter),
      is_this_struct_satisfy_something(turn_on_something) {

    bool assume_to_be_false = false;
    if (turn_on_something == assume_to_be_false) {
      throw Error("The construction of your State does not satisfy something");
    }
  }

  //To enable generic printer you need to implement these functions
  inline
  arma::vec positional_expectation() {
    // your specific implementation to report the expectations for real space positions
  }
  inline
  arma::vec momentum_expectation() {
    // your specific implementation to report the expectations for momentum
  }
};

}

#endif //METHOD_METHOD_TEMPLATE_H
