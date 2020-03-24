#ifndef EXE_UTIL_PTREE_H
#define EXE_UTIL_PTREE_H

#include <armadillo>
#include <boost/property_tree/ptree.hpp>
#include <quartz>

namespace quartz {
namespace util {

namespace ptree = boost::property_tree;

template<typename T>
std::vector<T> get_list(const ptree::ptree & pt) {

  std::vector<T> result;

  for (const auto & unit : pt) {

    if (unit.first != "element") {
      continue;
    }

    const auto value = unit.second.get_value_optional<T>();
    if (!value) {
      throw Error("Error reading value");
    }
    result.push_back(*value);
  }

  return result;
}


template<typename T>
arma::Mat<T> get_mat(const ptree::ptree & list_tree) {

  arma::Mat<T> result;

  for (const auto & line : list_tree) {
    if (line.first != "element") {
      continue;
    }
    const auto list_elements = arma::Col<T>(get_list<T>(line.second));
    arma::join_cols(result, list_elements.t());
  }

  return result;

}


template<typename T>
arma::Cube<T> get_cube(const ptree::ptree & list_tree) {

  arma::Cube<T> result;

  for (const auto & line1 : list_tree) {
    if (line1.first != "element") {
      continue;
    }

    const auto list_of_vecs = get_mat<T>(line1.second);
    arma::join_slices(result, list_of_vecs);
  }

  return result;
}


}
}

#endif //EXE_UTIL_PTREE_H
