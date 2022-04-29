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

void grid_1() {
  Triangulation<2> triangulation;

  GridIn<2> grid_in;
  grid_in.attach_triangulation(triangulation);

  std::ifstream f("example.msh");
  grid_in.read_msh(f);

  print_mesh_info(triangulation, "grid-1.vtu");
}

void grid_2() {
  Triangulation<2> tria1;
  GridGenerator::hyper_cube_with_cylindrical_hole(tria1, 0.25, 1.0);

  Triangulation<2> tria2;
  std::vector<unsigned int> repetitions(2);
  repetitions[0] = 3;
  repetitions[1] = 2;
  GridGenerator::subdivided_hyper_rectangle(
      tria2, repetitions, Point<2>(1.0, -1.0), Point<2>(4.0, 1.0));
  Triangulation<2> triangulation;
  GridGenerator::merge_triangulations(tria1, tria2, triangulation);

  print_mesh_info(triangulation, "grid-2.vtu");
}

void grid_3() {
  Triangulation<2> triangulation;
  GridGenerator::hyper_cube_with_cylindrical_hole(triangulation, 0.25, 1.0);
  print_mesh_info(triangulation, "grid-3-pre.vtu");

  for (const auto &cell : triangulation.active_cell_iterators()) {
    for (const auto i : cell->vertex_indices()) {
      Point<2> &v = cell->vertex(i);
      if (std::abs(v(1) - 1.0) < 1e-5) v(1) += 0.5;
    }
  }

  print_mesh_info(triangulation, "grid-3.vtu");
}

void grid_4() {
  Triangulation<2> triangulation;
  Triangulation<3> out;
  GridGenerator::hyper_cube_with_cylindrical_hole(triangulation, 0.25, 1.0);
  GridGenerator::extrude_triangulation(triangulation, 5, 2, out);
  print_mesh_info(out, "grid-4.vtu");
}

int main(int argc, char **argv) {
  USE(argc)
  USE(argv)

  try {
    grid_1();
    grid_2();
    grid_3();
    grid_4();
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
