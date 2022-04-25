#define USE(X) (void)(X);

int main(int argc, char **argv) {
  USE(argc)
  USE(argv)
  return 0;
}
