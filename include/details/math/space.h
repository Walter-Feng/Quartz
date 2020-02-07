#ifndef MATH_SPACE_H
#define MATH_SPACE_H

namespace math {
namespace space {

inline
arma::uword
indices_to_index(const arma::uvec & indices, const arma::uvec & table) {

  assert(indices.n_elem == table.n_elem);

  const arma::uword dims = table.n_elem;

  arma::uword index = 0;

  for (arma::uword i = 0; i < dims; i++) {
    index += table(i) * indices(i);
  }

  return index;
}

inline
arma::uvec index_to_indices(const arma::uword index, const arma::uvec & table) {

  const arma::uword dims = table.n_elem;

  arma::uvec indices = arma::uvec(dims, arma::fill::zeros);

  arma::uword i = 0;

  arma::uword downgraded_index = index;

  for (i = 0; i < dims - 1; i++) {
    if (index < table(i)) break;
  }

  for (arma::uword j = i; j > 0; j--) {
    indices(j) = downgraded_index / table(j);
    downgraded_index = downgraded_index % table(j);
  }
  indices(0) = downgraded_index / table(0);

  return indices;
}

inline
arma::uvec grids_to_table(const arma::uvec & grids) {
  const arma::uword dims = grids.n_elem;

  arma::uword table_index = 1;

  arma::uvec table = arma::uvec(dims);

  for (arma::uword i = 0; i < dims; i++) {
    table(i) = table_index;
    table_index *= grids(i);
  }

  return table;
}

inline
arma::umat auto_iteration_over_dims(const arma::uvec & grid) {

  const arma::uvec table = grids_to_table(grid);

  const auto n_elem = arma::prod(grid);
  const auto dim = grid.n_elem;

  arma::umat result = arma::umat(dim, n_elem);

  #pragma omp parallel for
  for (arma::uword i = 0; i < n_elem; i++) {
    result.col(i) = index_to_indices(i, table);
  }

  return result;
}

inline
arma::mat points_generate(const arma::uvec & grids,
                           const arma::mat & begin_end_list) {

  const arma::vec begin_list = begin_end_list.col(0);
  const arma::vec end_list = begin_end_list.col(1);

  const arma::vec diff = end_list - begin_list;

  assert(grids.n_elem == begin_end_list.n_rows);

  const arma::vec steps = diff / (grids - arma::ones(grids.n_elem));

  const auto iterations = space::auto_iteration_over_dims(grids);

  arma::mat centers = arma::mat();

  #pragma omp parallel for
  for (arma::uword i = 0; i < iterations.n_cols; i++) {
    centers = arma::join_rows(centers, iterations.col(i) % steps + begin_list);
  }

  return centers;
}

}
}

#endif //MATH_SPACE_H
