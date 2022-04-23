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

template <int dim> Step4<dim>::Step4() : fe(2), dof_handler(triangulation) {}

template <int dim> void Step4<dim>::make_grid() {
  GridGenerator::hyper_cube<dim>(triangulation, -1, 1);
  triangulation.refine_global(5);
}

template <int dim> void Step4<dim>::setup_system() {}
