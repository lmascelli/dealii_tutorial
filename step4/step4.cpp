#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <fstream>
#include <iostream>

using namespace dealii;

#define USE(X)                                                                 \
  (void)(X); // this macro does nothing, it's just for
             // disable unused variable warnings

template <int dim> class Step4 {
public:
  Step4();
  void run();

private:
  void make_grid();
  void setup_system();
  void assemble_system();
  void solve();
  void output_results() const;

  Triangulation<dim> triangulation;
  FE_Q<dim> fe;
  DoFHandler<dim> dof_handler;

  SparsityPattern sparsity_pattern;
  SparseMatrix<double> system_matrix;

  Vector<double> solution;
  Vector<double> system_rhs;
};

template <int dim> class RightHandSide : public Function<dim> {
public:
  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0) const override;
};

template <int dim> class BoundaryValues : public Function<dim> {
public:
  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0) const override;
};

int main(int argc, char **argv) {
  USE(argc);
  USE(argv);
  return 0;
}

template <int dim>
double RightHandSide<dim>::value(const Point<dim> &p,
                                 const unsigned int component) const {
  USE(component);
  double return_value = 0.0;
  for (unsigned int i = 0; i < dim; ++i)
    return_value += 4.0 + std::pow(p(i), 4.0);
  return return_value;
}

template <int dim>
double BoundaryValues<dim>::value(const Point<dim> &p,
                                  const unsigned int component) const {
  USE(component);
  return p.square();
}

template <int dim> Step4<dim>::Step4() : fe(1), dof_handler(triangulation) {}

template <int dim> void Step4<dim>::make_grid() {
  GridGenerator::hyper_cube<dim>(triangulation, -1, 1);
  triangulation.refine_global(5);
}

template <int dim> void Step4<dim>::setup_system() {
  dof_handler.distribute_dofs(fe);
  DynamicSparsityPattern dynamic_sparsity_pattern(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dynamic_sparsity_pattern);
  sparsity_pattern.copy_from(dynamic_sparsity_pattern);

  system_matrix.reinit(sparsity_pattern);

  solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());
}

template <int dim> void Step4<dim>::assemble_system() {
  QGauss<dim> quadrature_formula(fe.degree + 1);
  RightHandSide<dim> right_hand_side;

  FEValues<dim> fe_values(fe, quadrature_formula,
                          update_values | dealii::update_gradients |
                              dealii::update_quadrature_points |
                              dealii::update_JxW_values);
  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators()) {
    fe_values.reinit(cell);
    cell_matrix = 0;

    for (const unsigned int q_index : fe_values.quadrature_point_indices())
      for (const unsigned int i : fe_values.dof_indices()) {
        for (const unsigned int j : fe_values.dof_indices())
          cell_matrix(i, j) +=
              (fe_values.shape_grad(i, q_index) *
               fe_values.shape_grad(j, q_index) * fe_values.JxW(q_index));
        const auto &x_q = fe_values.quadrature_point(q_index);
        cell_rhs(i) += (fe_values.shape_value(i, q_index) *
                        right_hand_side.value(x_q) * fe_values.JxW(q_index));
      }

    cell->get_dof_indices(local_dof_indices);
    for (const unsigned int i : fe_values.dof_indices()) {
      for (const unsigned int j : fe_values.dof_indices()) {
        system_matrix.add(local_dof_indices[i], local_dof_indices[j],
                          cell_matrix(i, j));
      }
      system_rhs(local_dof_indices[i]) += cell_rhs(i);
    }
  }

  std::map<types::global_dof_index, double> boundary_values;
  VectorTools::interpolate_boundary_values(
      dof_handler, 0, BoundaryValues<dim>(), boundary_values);
  MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution,
                                     system_rhs);
}
