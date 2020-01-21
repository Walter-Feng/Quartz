// name your method
#ifndef METHOD_DVR_H
#define METHOD_DVR_H

// include only the necessary header files
#include "propagate.h"
#include "details/math/polynomial.h"
#include "details/math/constants.h"
#include "details/math/space.h"

#include "util/member_function_wrapper.h"

namespace quartz {
namespace method {
// use your method name to create a subspace for your
// implementation of details
namespace dvr {

namespace details {
cx_double kinetic_matrix_element(const long long i,
                                 const long long j,
                                 const double interval,
                                 const double mass) {
  if (i == j) {
    return cx_double{
        math::pi * math::pi / 6. / mass / interval / interval, 0.};
  } else {
    return cx_double{
        std::pow(-1, i - j) / mass / interval / interval / (double) (i - j) /
        (double) (i - j), 0.};
  }
}

cx_double momentum_matrix_element(const long long i,
                                  const long long j,
                                  const double interval) {

  if (i == j) {
    return cx_double{0., 0.};
  } else {
    return -cx_double{0., std::pow(-1, i - j) / interval / (i - j)};
  }
}

}

} // namespace method_template

// It is suggested to satisfy both complex (cx_double) and real (double)
// and suitable for arbitrary dimensions
template<typename T>
struct DVRState {
public:
  arma::Col<T> coefs;
  arma::uvec grid;
  arma::mat ranges;
  arma::vec masses;

  // Establish an easy way to construct your State
  template<typename Wavefunction>
  DVRState(const Wavefunction & initial,
           const arma::uvec & grid,
           const arma::mat & range,
           const arma::vec & masses) :
           coefs(at(initial, math::space::points_generate(grid,range))),
           grid(grid),
           ranges(ranges),
           masses(masses)
           {
    if (grid.n_rows != ranges.n_rows) {
      throw Error("Different dimension between the grid and the range");
    }
   if (grid.n_rows != masses.n_rows) {
     throw Error("Different dimension between the grid and the masses");
   }
  }

  inline
  DVRState(const Wavefunction & initial,
           const arma::mat & points,
           const arma::vec & masses) :
      coefs(at(initial, math::space::points_generate(grid,range))),
      grid(grid),
      ranges(ranges),
      masses(masses)
  {
    if (grid.n_rows != ranges.n_rows) {
      throw Error("Different dimension between the grid and the range");
    }
    if (grid.n_rows != masses.n_rows) {
      throw Error("Different dimension between the grid and the masses");
    }
  }

  inline
  arma::Mat<T> kinetic_energy_matrix() {

  }

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
}

#endif //METHOD_DVR_H