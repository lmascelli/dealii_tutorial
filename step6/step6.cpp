#define USE(X) (void)(X);

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>

using namespace dealii;

template <int dim>
class Step6 {
 public:
  Step6();
  void run();

 private:
  void setup_system();
  void assemble_system();
  void solve();
  void refine_grid();
  void output_results(const unsigned int cycle) const;

  Triangulation<dim> triangulation;
  FE_Q<dim> fe;
  DoFHandler<dim> dof_handler;

  AffineConstraints<double> constraints;
  SparseMatrix<double> system_matrix;
  SparsityPattern sparsity_pattern;

  Vector<double> solution;
  Vector<double> system_rhs;
};

template <int dim>
double coefficient(const Point<dim> &p) {
  if (p.square() < 0.5 * 0.5)
    return 20;
  else
    return 1;
}

int main(int argc, char **argv) {
  USE(argc)
  USE(argv)
  return 0;
}

template <int dim>
Step6<dim>::Step6() : fe(2), dof_handler(triangulation) {}

template <int dim>
void Step6<dim>::setup_system() {
  dof_handler.distribute_dofs(fe);

  // TODO try to move all reinit at bottom
  solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());

  constraints.clear();
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  VectorTools::interpolate_boundary_values(
      dof_handler, 0, Functions::ZeroFunction<dim>(), constraints);
  constraints.close();

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);

  sparsity_pattern.copy_from(dsp);
  system_matrix.reinit(sparsity_pattern);
}

template <int dim>
void Step6<dim>::assemble_system() {
  const QGauss<dim> quadrature_formula(fe.degree + 1);
  FEValues<dim> fe_values(fe, quadrature_formula,
                          update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators()) {
    cell_matrix = 0;
    cell_rhs = 0;

    fe_values.reinit(cell);

    for (const unsigned int q_index : fe_values.quadrature_point_indices()) {
      const double current_coefficient =
          coefficient(fe_values.quadrature_point(q_index));
      for (const unsigned int i : fe_values.dof_indices()) {
        for (const unsigned int j : fe_values.dof_indices())
          cell_matrix(i, j) +=
              (current_coefficient * fe_values.shape_grad(i, q_index) *
               fe_values.shape_grad(j, q_index) * fe_values.JxW(q_index));
        cell_rhs(i) +=
            (fe_values.shape_value(i, q_index) * 1.0 * fe_values.JxW(q_index));
      }
    }

    cell->get_dof_indices(local_dof_indices);
    constraints.distribute_local_to_global(
        cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
  }
}

template <int dim>
void Step6<dim>::solve() {
  SolverControl solver_control(1000, 1e-12);
  SolverCG<Vector<double>> solver(solver_control);

  PreconditionSSOR<SparseMatrix<double>> preconditioner;
  preconditioner.initialize(system_matrix, 1.2);

  solver.solve(system_matrix, solution, system_rhs, preconditioner);

  constraints.distribute(solution);
}

template <int dim>
void Step6<dim>::refine_grid() {}
