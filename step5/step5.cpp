#define USE(X) (void)(X);

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/quadrature.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>
using namespace dealii;

template <int dim> class Step5 {
public:
  Step5();
  void run();

private:
  void setup_system();
  void assemble_system();
  void solve();
  void output_results(const unsigned int cycle) const;

  Triangulation<dim> triangulation;
  FE_Q<dim> fe;
  DoFHandler<dim> dof_handler;

  SparsityPattern sparsity_pattern;
  SparseMatrix<double> system_matrix;
  Vector<double> system_rhs;
  Vector<double> solution;
};

int main(int argc, char **argv) {
  USE(argc)
  USE(argv)
  return 0;
}

template <int dim> Step5<dim>::Step5() : fe(1), dof_handler(triangulation) {}

template <int dim> void Step5<dim>::run() {
  const unsigned int N_CYCLE = 5;
  for (unsigned int cycle = 0; cycle < N_CYCLE; ++cycle) {
    setup_system();
    assemble_system();
    solve();
    output_results(cycle);
  }
}

template <int dim> double coefficient(const Point<dim> &p) {
  if (p.square() < 0.5 * 0.5)
    return 20;
  else
    return 1;
}
