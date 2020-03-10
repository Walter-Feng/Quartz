#include <Catch2/catch.hpp>

#include <quartz>

namespace quartz {

TEST_CASE("Propagate") {

  std::cout << std::endl << std::endl;
  std::cout << "testing propagate function ..." << std::endl << std::endl;
  SECTION("Schrotinger") {

    std::cout << std::endl <<
              "This is " << "DVR" << std::endl;

    const double dt = 0.01;

    const method::dvr::State initial_state =
        method::dvr::State(math::Gaussian<double>(arma::mat{1.}, arma::vec{1}),
                           arma::uvec{100},
                           arma::mat{{-5, 5}});

    const auto harmonic_potential = math::Polynomial<double>(arma::vec{0.5},
                                                             lmat{2});

    const auto op = method::dvr::Operator(initial_state, harmonic_potential);

    const auto wrapper =
        math::schrotinger_wrapper<method::dvr::Operator,
                                  method::dvr::State,
                                  math::Polynomial<double>>;

    const auto result = propagate(initial_state,
                                  op,
                                  wrapper,
                                  harmonic_potential,
                                  generic_printer<method::dvr::State>, 10, dt,
                                  2);


  }

  SECTION("Classical Wigner Approximation") {

    std::cout << std::endl <<
              "This is " << "Classical Wigner Approximation" << std::endl;

    const double dt = 0.01;

    const method::md::State initial_state =
        method::md::State(math::Gaussian<double>(arma::mat{1.}, arma::vec{1}).wigner_transform(),
                          arma::uvec{30,30},
                          arma::mat{{-5, 5},{-5,5}});

    const auto harmonic_potential = math::Polynomial<double>(arma::vec{0.5},
                                                             lmat{2});

    const auto op = method::md::Operator(initial_state, harmonic_potential);

    const auto wrapper =
        math::runge_kutta_4<method::md::Operator<math::Polynomial<double>>,
            method::md::State,
            math::Polynomial<double>>;

    const auto result = propagate(initial_state,
                                  op,
                                  wrapper,
                                  harmonic_potential,
                                  generic_printer<method::md::State>, 10, dt,
                                  2);


  }

  SECTION("Wave Packet") {

    std::cout << std::endl <<
              "This is " << "Wave Packet" << std::endl;

    const double dt = 0.01;

    const auto initial_state =
        method::packet::State(math::Gaussian<double>(arma::mat{1.}, arma::vec{1}).wigner_transform(),
                              arma::uvec{20,20},
                              arma::mat{{-10, 10},{-10,10}});

    const auto harmonic_potential = math::Polynomial<double>(arma::vec{0.5},
                                                             lmat{2});

    const auto op = method::packet::Operator(initial_state, harmonic_potential);

    const auto wrapper =
        math::runge_kutta_4<method::packet::Operator,
            method::packet::State,
            math::Polynomial<double>>;

    const auto result = propagate(initial_state,
                                  op,
                                  wrapper,
                                  harmonic_potential,
                                  generic_printer<method::packet::State>, 10, dt,
                                  2);


  }

  SECTION("Fixed Gaussian Basis") {

    std::cout << std::endl <<
              "This is " << "Fixed Gaussian Basis" << std::endl;

    const double dt = 0.01;

    const auto initial_state =
        method::fgb::State(math::Gaussian<double>(arma::mat{1.}, arma::vec{1}).wigner_transform(),
                              arma::uvec{10,10},
                              arma::mat{{-3, 3},{-3,3}},
                              1);

    const auto harmonic_potential = math::Polynomial<double>(arma::vec{0.5},
                                                             lmat{2});

    const auto op = method::fgb::Operator(initial_state, harmonic_potential);

    const auto wrapper =
        math::runge_kutta_4<method::fgb::Operator,
            method::fgb::State,
            math::Polynomial<double>>;

    const auto result = propagate(initial_state,
                                  op,
                                  wrapper,
                                  harmonic_potential,
                                  generic_printer<method::fgb::State>, 10, dt,
                                  2);


  }

  SECTION("Wigner Dynamics") {

    std::cout << std::endl <<
              "This is " << "Wigner Dynamics" << std::endl;

    const double dt = 0.01;

    const method::wd::State initial_state =
        method::wd::State(math::Gaussian<double>(arma::mat{1.}, arma::vec{1}).wigner_transform(),
                          arma::uvec{30,30},
                          arma::mat{{-5, 5},{-5,5}});

    const auto harmonic_potential = math::Polynomial<double>(arma::vec{0.5},
                                                             lmat{2});

    const auto wrapped_initial =
        math::GaussianWithPoly(
            math::Gaussian<double>(arma::mat{1.}, arma::vec{1}).wigner_transform()
            );

    const auto op = method::wd::Operator(initial_state, wrapped_initial, harmonic_potential);

    const auto wrapper =
        math::runge_kutta_4<method::wd::Operator<math::Polynomial<double>,
                            math::GaussianWithPoly<double>>,
            method::wd::State,
            math::Polynomial<double>>;

    const auto result = propagate(initial_state,
                                  op,
                                  wrapper,
                                  harmonic_potential,
                                  generic_printer<method::wd::State>, 10, dt,
                                  2);


  }

  SECTION("Donoso-Martens Dynamics") {

    std::cout << std::endl <<
              "This is " << "Donoso-Martens Dynamics" << std::endl;

    const double dt = 0.01;

    const method::dmd::State initial_state =
        method::dmd::State(math::Gaussian<double>(arma::mat{1.}, arma::vec{1}).wigner_transform(),
                          arma::uvec{30,30},
                          arma::mat{{-5, 5},{-5,5}});

    const auto harmonic_potential = math::Polynomial<double>(arma::vec{0.5},
                                                             lmat{2});

    const auto op = method::dmd::Operator(initial_state, harmonic_potential);

    const auto wrapper =
        math::runge_kutta_4<method::dmd::Operator<math::Polynomial<double>>,
            method::dmd::State,
            math::Polynomial<double>>;

    const auto result = propagate(initial_state,
                                  op,
                                  wrapper,
                                  harmonic_potential,
                                  generic_printer<method::dmd::State>, 10, dt,
                                  2);


  }

  SECTION("Classical Wigner Approximation + Semi Moyal Dynamics") {

    std::cout << std::endl <<
    "This is " << "Classical Wigner Approximation + Semi Moyal Dynamics" << std::endl;
    const double dt = 0.01;

    const method::cwa_smd::State initial_state =
        method::cwa_smd::State(math::Gaussian<double>(arma::mat{1.}, arma::vec{1}).wigner_transform(),
                           arma::uvec{30,30},
                           arma::mat{{-5, 5},{-5,5}},3);

    const auto harmonic_potential = math::Polynomial<double>(arma::vec{0.5},
                                                             lmat{2});

    const auto op = method::cwa_smd::Operator(initial_state, harmonic_potential);

    const auto wrapper =
        math::runge_kutta_4<method::cwa_smd::Operator,
            method::cwa_smd::State,
            math::Polynomial<double>>;

    const auto result = propagate(initial_state,
                                  op,
                                  wrapper,
                                  harmonic_potential,
                                  generic_printer<method::cwa_smd::State>, 10, dt,
                                  2);


  }
}

}
