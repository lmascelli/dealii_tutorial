#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>
#include <fstream>
#include <iostream>
#include <ostream>
using namespace dealii;

void first_grid() {
  Triangulation<2> triangulation;
  GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global(4);
  std::ofstream out("grid-1.svg");
  GridOut grid_out;
  grid_out.write_svg(triangulation, out);
  std::cout << "Grid written to grid-1.svg" << std::endl;
}

void second_grid() {
  Triangulation<2> triangulation;
  const Point<2> center(1, 0);
  const double inner_radius = 0.5, outer_radius = 1.0;
  GridGenerator::hyper_shell(triangulation, center, inner_radius, outer_radius,
                             10);
  for (unsigned int step = 0; step < 5; ++step) {
    for (auto &cell : triangulation.active_cell_iterators()) {
      for (auto v : cell->vertex_indices()) {
        const double distance_from_center = center.distance(cell->vertex(v));
        if (std::fabs(distance_from_center - inner_radius) <=
            1e-6 * inner_radius) {
          cell->set_refine_flag();
          break;
        }
      }
    }
    triangulation.execute_coarsening_and_refinement();
  }

  std::ofstream out("grid-2.svg");
  GridOut grid_out;
  grid_out.write_svg(triangulation, out);
  std::cout << "Grid written to grid-2.svg" << std::endl;
}

int main(int argc, char **argv) {
  first_grid();
  second_grid();
  return 0;
}
