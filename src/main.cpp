#include <quartz>
#include <args.hxx>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include "run.h"
#include "parse/math/polynomial.h"
#include "parse/math/gaussian.h"
#include "parse/methods/cwa.h"

int main(const int argc, const char * argv[]) {

  using namespace quartz;
  namespace ptree = boost::property_tree;

  args::ArgumentParser parser("This is a simple quartz interface "
                              "that helps you construct quick simulation "
                              "of polynomial potentials.");

  args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});

  args::ValueFlag<std::string> model_flag(parser, "model",
                                          "A specific model you would like to try."
                                          "This option will be neglected if input file is "
                                          "specified. "
                                          "Currently it enables direct input of: \n"
                                          "harmonic(default) - for harmonic oscillator \n"
                                          "anharmonic - for anharmonic oscillator \n"
                                          "double_well - for double well model, and will set mass "
                                          "automatically to 1836, dt to 1 ",
                                          {"model"});

  args::ValueFlag<std::string> method_flag(parser, "method",
                                           "The method you will be going to test. "
                                           "This option will be neglected if input file is "
                                           "specified.\n"
                                           "Currently supported methods: \n"
                                           "DVR(default), \n"
                                           "CWA, \n"
                                           "CWA_SMD\n"
                                           "DVR_SMD\n",
                                           {"method"});

  args::ValueFlag<arma::uword> grid_flag(parser, "grid",
                                         "The grid size for a specific method. "
                                         "Default setting is 30.",
                                         {"grid"});

  args::ValueFlag<double> mass_flag(parser, "mass",
                                    "The mass for the particle. "
                                    "Default setting is 1.", {"mass"});

  args::ValueFlag<double> dt_flag(parser, "time step",
                                  "The time step for simulation. "
                                  "Default setting is 0.01 a.u.",
                                  {"dt"});

  args::ValueFlag<double> span_flag(parser, "span",
                                    "The area [-span, span] for "
                                    "the initialisation of simulation. "
                                    "Default setting is [-5, 5]",
                                    {"span"});

  args::ValueFlag<arma::uword> step_flag(parser, "Number of steps",
                                         "The total number of steps for simulation. "
                                         "Default setting is 1000 steps",
                                         {"step"});

  args::ValueFlag<int> print_level_flag(parser, "Print Level",
                                        "The print level for output to screen. "
                                        "Default is 2.",
                                        {'p', "print_level"}
  );

//  TODO: enable input when introducing boost::property_tree
  args::Positional<std::string> input_flag(parser, "input",
                                           "The input file (in json format)");

  args::PositionalList<double> options(parser, "options",
                                       "The options some of the methods would require. "
                                       "Requirements:\n"
                                       "CWA_SMD: [grade(int)]\n"
                                       "");
  try {
    parser.ParseCLI(argc, argv);
  }
  catch (const args::Help &) {
    std::cout << parser << std::endl;
    return 0;
  }
  catch (const args::ParseError & e) {
    std::cout << e.what() << std::endl;
    std::cout << parser << std::endl;
    return 1;
  }
  catch (const args::ValidationError & e) {
    std::cout << e.what() << std::endl;
    std::cout << parser << std::endl;
    return 1;
  }

  ///////////////////// Read Input File /////////////////////

  if (input_flag) {

    ptree::ptree input;

    ptree::read_json(args::get(input_flag), input);

    return run(input);

  }

    ///////////////////// Global Parameters /////////////////////

    arma::uword grid_size = 30;
  if (grid_flag) {
    grid_size = args::get(grid_flag);
  }

  arma::uword steps = 1000;
  double mass = 1;
  double dt = 0.01;
  double span = 5;

  int print_level = 2;

  auto initial_wf = math::Gaussian<cx_double>(arma::mat{1.}, arma::cx_vec{1});

  ///////////////////// Potential (model) //////////////////////
  auto potential = math::Polynomial<double>(arma::vec{0.5}, lmat{2});

  if (model_flag) {
    const std::string model = args::get(model_flag);

    if (model == "harmonic") {}
    if (model == "anharmonic") {
      throw Error("anharmonic potential not implemented");
    }
    if (model == "double_well") {
      potential =
          math::Polynomial<double>(arma::vec{-0.0003, 0.000024}, lmat{{2, 4}});
      mass = 1836;
      dt = 1;
      initial_wf =
          math::Gaussian<double>(arma::mat{0.5}, arma::vec{-2.5})
              .with_phase_factor(arma::vec{2});
    }
  }

  ///////////////////// Check parameter flag /////////////////////
  if (mass_flag) { mass = args::get(mass_flag); }
  if (dt_flag) { dt = args::get(dt_flag); }
  if (span_flag) { span = args::get(span_flag); }
  if (step_flag) { steps = args::get(step_flag); }
  if (print_level_flag) { print_level = args::get(print_level_flag); }

  ///////////////////// Method //////////////////////
  if (method_flag) {
    const std::string method = args::get(method_flag);

    if (method == "dvr") {
      const method::dvr::State initial_state =
          method::dvr::State(initial_wf,
                             arma::uvec{grid_size},
                             arma::mat{{-span, span}},
                             arma::vec{mass});

      const auto op = method::dvr::Operator(initial_state, potential);

      const auto wrapper =
          math::schrotinger_wrapper<method::dvr::Operator,
              method::dvr::State,
              math::Polynomial<double>>;

      const auto result = propagate(initial_state, op, wrapper, potential,
                                    generic_printer<method::dvr::State>,
                                    steps, dt, print_level);
    } else if (method == "cwa") {
      const method::cwa::State initial_state =
          method::cwa::State(initial_wf.wigner_transform(),
                             arma::uvec{grid_size, grid_size},
                             arma::mat{{-span, span},
                                      {-span, span}},
                             arma::vec{mass});

      const auto op = method::cwa::Operator(initial_state, potential);

      const auto wrapper =
          math::runge_kutta_4<method::cwa::Operator<math::Polynomial<double>>,
              method::cwa::State,
              math::Polynomial<double>>;

      const auto result = propagate(initial_state, op, wrapper, potential,
                                    generic_printer<method::cwa::State>,
                                    steps, dt, print_level);
    } else if (method == "cwa_smd") {
      if (!options) {
        throw Error("grade for cwa_smd not specified");
      }
      const int grade = args::get(options)[0] + 1;
      const method::cwa_smd::State initial_state =
          method::cwa_smd::State(initial_wf.wigner_transform(),
                                 arma::uvec{grid_size, grid_size},
                                 arma::mat{{-span, span},
                                           {-span, span}},
                                 arma::vec{mass}, grade);

      const auto op = method::cwa_smd::Operator(initial_state, potential);

      const auto wrapper =
          math::runge_kutta_4<method::cwa_smd::Operator,
              method::cwa_smd::State,
              math::Polynomial<double>>;

      const auto result = propagate(initial_state, op, wrapper, potential,
                                    generic_printer<method::cwa_smd::State>,
                                    steps, dt, print_level);
    } else {
      throw Error("Method " + method + " is not supported.");
    }
  } else { // perform DVR
    const method::dvr::State initial_state =
        method::dvr::State(initial_wf,
                           arma::uvec{grid_size},
                           arma::mat{{-span, span}},
                           arma::vec{mass});

    const auto op = method::dvr::Operator(initial_state, potential);

    const auto wrapper =
        math::schrotinger_wrapper<method::dvr::Operator,
            method::dvr::State,
            math::Polynomial<double>>;

    const auto result = propagate(initial_state, op, wrapper, potential,
                                  generic_printer<method::dvr::State>,
                                  steps, dt, print_level);
  }

  return 0;
}
