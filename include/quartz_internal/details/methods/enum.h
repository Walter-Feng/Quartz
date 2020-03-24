#ifndef QUARTZ_ENUM_H
#define QUARTZ_ENUM_H

#include "dvr.h"
#include "cwa.h"
#include "cwa_smd.h"
#include "dvr_smd.h"

enum Methods {
  DVR,
  CWA,
  CWA_SMD,
  DVR_SMD
};

using State = std::variant<
    method::dvr::State,
    method::cwa::State,
    method::cwa_smd::State,
    method::dvr_smd::State
    >;

const std::map<std::string, Methods>
    method_map = {
    {"dvr", Methods::DVR},
    {"cwa", Methods::CWA},
    {"cwa_smd", Methods::CWA_SMD},
    {"dvr_smd", Methods::DVR_SMD}
};

#endif //QUARTZ_ENUM_H
