#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/lac/sparsity_pattern.h>
#include <fstream>

using namespace dealii;

void make_grid(Triangulation<2> &triangulation) {
  const Point<2> center(1, 0);
  const double inner_radius = 0.5, outer_radius = 1.0;
  GridGenerator::hyper_shell(triangulation, center, inner_radius, outer_radius);

  for (unsigned int step = 0; step < 3; ++step) {
    for (auto &cell : triangulation.active_cell_iterators())
      for (const auto v : cell->vertex_indices()) {
        const double distance_from_center = center.distance(cell->vertex(v));
        if (std::fabs(distance_from_center - inner_radius) <=
            inner_radius * 1e-6) {
          cell->set_refine_flag();
          break;
        }
      }
    triangulation.execute_coarsening_and_refinement();
  }
}

void distribute_dofs(DoFHandler<2> &dof_handler) {
  const FE_Q<2> finite_element(1);
  dof_handler.distribute_dofs(finite_element);
  SparsityPattern sparsity_pattern;
  DynamicSparsityPattern dynamic_sparsity_pattern(dof_handler.n_dofs(),
                                                  dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dynamic_sparsity_pattern);
  sparsity_pattern.copy_from(dynamic_sparsity_pattern);
  std::ofstream out("sparsity_pattern1.svg");
  sparsity_pattern.print_svg(out);
}

void renumbering_dofs(DoFHandler<2> &dof_handler) {
  DoFRenumbering::Cuthill_McKee(dof_handler);
  DynamicSparsityPattern dynamic_sparsity_pattern(dof_handler.n_dofs(),
                                                  dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dynamic_sparsity_pattern);
  SparsityPattern sparsity_pattern;
  sparsity_pattern.copy_from(dynamic_sparsity_pattern);
  std::ofstream out("sparsity_pattern2.svg");
  sparsity_pattern.print_svg(out);
}

int main(int argc, char **argv) {
  Triangulation<2> triangulation;
  make_grid(triangulation);

  DoFHandler<2> dof_handler(triangulation);

  distribute_dofs(dof_handler);
  renumbering_dofs(dof_handler);
  return 0;
}