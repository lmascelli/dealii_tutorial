/*
 * THE SYSTEM
 *
 * du/dt - v = 0
 * dv/dt - lamda(u) = f
 * u(x, t) = g on dSigma
 * u(x, 0) = u0(x) on Sigma
 * v(x, 0) = u1(x) on Sigma
 *
 * TIME DISCRETIZATION
 *
 * k = t;n - t;n-1
 *
 * (u;n - u;n-1)/k - [sigma * v;n + (1-sigma) * v;n-1] = 0
 * (v;n - v;n-1)/k - [sigma * lambda(u);n + (1-sigma) * lambda(u);n-1] = sigma *
 *                                                    f;n + (1 - sigma) * f;n-1
 *
 * with sigma = 0.5 --> Crank-Nicolson method
 *
 * v;n = v;n-1 + k(sigma * (f;n + lambda(u);n) + (1-sigma) * (lambda(u);n-1 +
 *                                                            f;n-1))
 *
 * (u;n - u;n-1)/k =
 *      sigma * v;n-1 +
 *      sigma * k * (sigma* (f;n + lambda(u);n)) +
 *      sigma * k * (1 - sigma) * lambda(u); n-1 +
 *      (1-sigma) * v;n-1
 *
 *
 */

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>

namespace Step23 {
using namespace dealii;

template <int dim>
class WaveEquation {
 public:
  WaveEquation();
  void run();

 private:
  void setup_system();
  void solve_u();
  void solve_v();
  void output_results() const;

  Triangulation<dim> triangulation;
  FE_Q<dim> fe;
  DoFHandler<dim> dof_handler;

  AffineConstraints<double> constraints;

  SparsityPattern sparsity_pattern;
  SparseMatrix<double> mass_matrix;
  SparseMatrix<double> laplace_matrix;
  SparseMatrix<double> matrix_u;
  SparseMatrix<double> matrix_v;

  Vector<double> solution_u, solution_v;
  Vector<double> old_solution_u, old_solution_v;
  Vector<double> system_rhs;

  double time_step;
  double time;
  unsigned int timestep_number;
  const double theta;
};

template <int dim>
class InitialValuesU : public Function<dim> {
 public:
  virtual double value(const Point<dim> &,
                       const unsigned int component = 0) const override {
    (void)component;
    Assert(component == 0, ExcIndexRange(component, 0, 1));
    return 0;
  }
};

template <int dim>
class InitialValuesV : public Function<dim> {
 public:
  virtual double value(const Point<dim> &,
                       const unsigned int component = 0) const override {
    (void)component;
    Assert(component == 0, ExcIndexRange(component, 0, 1));
    return 0;
  }
};

template <int dim>
class RightHandSide : public Function<dim> {
 public:
  virtual double value(const Point<dim> &,
                       const unsigned int component = 0) const override {
    (void)component;
    Assert(component == 0, ExcIndexRange(component, 0, 1));
    return 0;
  }
};

template <int dim>
class BoundaryValuesU : public Function<dim> {
 public:
  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0) const override {
    (void)component;
    Assert(component == 0, ExcIndexRange(component, 0, 1));

    if ((this->get_time() <= 0.5) && (p[0] < 0) && (p[1] < 1. / 3) &&
        (p[1] > -1. / 3))
      return std::sin(this->get_time() * 4 * numbers::PI);
    else
      return 0;
  }
};

template <int dim>
class BoundaryValuesV : public Function<dim> {
 public:
  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0) const override {
    (void)component;
    Assert(component == 0, ExcIndexRange(component, 0, 1));

    if ((this->get_time() <= 0.5) && (p[0] < 0) && (p[1] < 1. / 3) &&
        (p[1] > -1. / 3))
      return (std::cos(this->get_time() * 4 * numbers::PI) * 4 * numbers::PI);
    else
      return 0;
  }
};

template <int dim>
WaveEquation<dim>::WaveEquation()
    : fe(1),
      dof_handler(triangulation),
      time_step(1. / 64),
      time(time_step),
      timestep_number(1),
      theta(0.5) {}

template <int dim>
void WaveEquation<dim>::setup_system() {
  GridGenerator::hyper_cube(triangulation, -1, 1);
  triangulation.refine_global(7);

  std::cout << "Number of active cells: "
            << triangulation.n_global_active_cells() << std::endl;
  dof_handler.distribute_dofs(fe);

  std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs()
            << std::endl
            << std::endl;

  DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);

  sparsity_pattern.copy_from(dsp);

  mass_matrix.reinit(sparsity_pattern);
  laplace_matrix.reinit(sparsity_pattern);
  matrix_u.reinit(sparsity_pattern);
  matrix_v.reinit(sparsity_pattern);

  MatrixCreator::create_mass_matrix(dof_handler, QGauss<dim>(fe.degree + 1),
                                    mass_matrix);
  MatrixCreator::create_laplace_matrix(dof_handler, QGauss<dim>(fe.degree + 1),
                                       laplace_matrix);

  solution_u.reinit(dof_handler.n_dofs());
  solution_v.reinit(dof_handler.n_dofs());
  old_solution_u.reinit(dof_handler.n_dofs());
  old_solution_v.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());

  constraints.close();
}

template <int dim>
void WaveEquation<dim>::solve_u() {
  SolverControl solver_control(1000, 1e-8 * system_rhs.l2_norm());
  SolverCG<Vector<double>> cg(solver_control);

  cg.solve(matrix_u, solution_u, system_rhs, PreconditionIdentity());

  std::cout << "  u-equation: " << solver_control.last_step()
            << "  CG iterations." << std::endl;
}

template <int dim>
void WaveEquation<dim>::solve_v() {
  SolverControl solver_control(1000, 1e-8 * system_rhs.l2_norm());
  SolverCG<Vector<double>> cg(solver_control);

  cg.solve(matrix_v, solution_v, system_rhs, PreconditionIdentity());

  std::cout << "  v-equation: " << solver_control.last_step()
            << "  CG iterations." << std::endl;
}

template <int dim>
void WaveEquation<dim>::output_results() const {
  DataOut<dim> data_out;

  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution_u, "U");
  data_out.add_data_vector(solution_v, "V");

  data_out.build_patches();

  const std::string filename =
      "solution-" + Utilities::int_to_string(timestep_number, 3) + ".vtu";

  DataOutBase::VtkFlags vtk_flags;
  vtk_flags.compression_level =
      DataOutBase::VtkFlags::ZlibCompressionLevel::best_speed;
  data_out.set_flags(vtk_flags);
  std::ofstream output(filename);
  data_out.write_vtu(output);
}

template <int dim>
void WaveEquation<dim>::run() {
  setup_system();

  VectorTools::project(dof_handler, constraints, QGauss<dim>(fe.degree + 1),
                       InitialValuesU<dim>(), old_solution_u);
  VectorTools::project(dof_handler, constraints, QGauss<dim>(fe.degree + 1),
                       InitialValuesV<dim>(), old_solution_v);
}
}  // namespace Step23

int main(int argc, char **argv) {
  try {
    using namespace Step23;

    WaveEquation<2> wave_equation_solver;
    wave_equation_solver.run();
  } catch (std::exception &exc) {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;

    return 1;

  } catch (...) {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }

  return 0;
}
