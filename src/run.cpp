#include "run.h"
#include "util/json_printer.h"

#include "util/ptree.h"
#include "util/json_parser.h"
#include "parse/math/gaussian.h"
#include "parse/math/polynomial.h"
#include "parse/math/math_object.h"
#include "parse/methods/cwa.h"
#include "parse/methods/cwa_smd.h"
#include "parse/methods/cwa_smd_opt.h"
#include "parse/methods/m_dvr.h"
#include "parse/methods/dvr.h"
#include "parse/methods/dvr_smd.h"
#include "parse/methods/g_dvr_smd.h"

namespace quartz {

ptree::ptree run(const ptree::ptree & input) {

  const std::string method = input.get("method", "dvr");

  ptree::ptree result;
  util::put<std::string>(result, "Library", "Quartz");

  if(method == "m_dvr") {

    result.put_child("result", m_dvr(input));

  }

  const auto initial = parse::gaussian(input.get_child("initial"));
  const auto potential = parse::math_object(input.get_child("potential"));

  result.put_child("initial", input.get_child("initial"));
  result.put_child("potential", input.get_child("potential"));

  if (method == "dvr") {

    result.put_child("result", dvr(input, potential, initial));

  } else if (method == "cwa") {

    result.put_child("result", cwa(input, potential, initial));

  } else if (method == "cwa_smd") {

    result.put_child("result", cwa_smd(input, potential, initial));

  } else if (method == "cwa_smd_opt") {

    result.put_child("result", cwa_smd_opt(input, potential, initial));

  } else if (method == "dvr_smd") {

    result.put_child("result", dvr_smd(input, potential, initial));

  } else if (method == "g_cwa_smd") {

    result.put_child("result", g_cwa_smd(input, potential, initial));

  } else {
    throw Error("The method specified is not implemented");
  }

  return result;
}

}