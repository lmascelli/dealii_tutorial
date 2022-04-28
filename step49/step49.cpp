#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>

#include <fstream>
#include <iostream>
#include <map>

using namespace dealii;

#define USE(X) (void)(X);

template <int dim>
void print_mesh_info(const Triangulation<dim> &triangulation,
                     const std::string &filename) {
  std::cout << "Mesh info:" << std::endl
            << "  dimension: " << dim << std::endl
            << "  no. of cells: " << triangulation.n_active_cells()
            << std::endl;
  {
    std::map<types::boundary_id, unsigned int> boundary_count;
    for (const auto &face : triangulation.active_face_iterators())
      if (face->at_boundary()) boundary_count[face->boundary_id()]++;

    std::cout << "  boundary indicators: ";
    for (const std::pair<const types::boundary_id, unsigned int> &pair :
         boundary_count) {
      std::cout << pair.first << "(" << pair.second << " times) ";
    }
    std::cout << std::endl;
  }
  std::ofstream out(filename);
  GridOut grid_out;
  grid_out.write_vtu(triangulation, out);
  std::cout << "  written to " << filename << std::endl << std::endl;
}

int main(int argc, char **argv) {
  USE(argc)
  USE(argv)

  return 0;
}
