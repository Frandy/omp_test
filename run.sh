
g++ omp_test.cpp -fopenmp -o default_omp -O3
./default_omp 2

g++ omp_test.cpp -I/home/chjd/Work/fastlib/bolt-omp/include -L/home/chjd/Work/fastlib/bolt-omp/lib -lbolt -L/home/chjd/Work/fastlib/bolt-abt/lib -labt -o bolt_omp -O3
./bolt_omp 2
