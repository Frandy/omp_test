
#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <random>
#include <thread>

#include <omp.h>

int N = 0;
double **a = nullptr, **b = nullptr, **c = nullptr;

std::chrono::microseconds measure_time_omp(unsigned);

inline void allocate_matrix() {
  a = static_cast<double **>(std::malloc(N * sizeof(double *)));
  b = static_cast<double **>(std::malloc(N * sizeof(double *)));
  c = static_cast<double **>(std::malloc(N * sizeof(double *)));
  for (int i = 0; i < N; ++i) {
    a[i] = static_cast<double *>(std::malloc(N * sizeof(double)));
    b[i] = static_cast<double *>(std::malloc(N * sizeof(double)));
    c[i] = static_cast<double *>(std::malloc(N * sizeof(double)));
  }
}

inline void deallocate_matrix() {
  for (int i = 0; i < N; ++i) {
    std::free(a[i]);
    std::free(b[i]);
    std::free(c[i]);
  }
  std::free(a);
  std::free(b);
  std::free(c);
}

inline int64_t reduce_sum() {
  int64_t sum{0};
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; ++j) {
      sum += c[i][j];
    }
  }
  return sum;
}

void matrix_multiplication(const std::string &model, const unsigned num_threads,
                           const unsigned num_rounds) {

  std::cout << std::setw(12) << "size" << std::setw(12) << "runtime"
            << std::endl;

  for (int i = 128; i <= 1024; i += 32) {

    N = i;

    allocate_matrix();

    double runtime{0.0};

    for (unsigned j = 0; j < num_rounds; ++j) {
      runtime += measure_time_omp(num_threads).count();
    }

    std::cout << std::setw(12) << N << std::setw(12)
              << runtime / num_rounds / 1e3 << std::endl;

    deallocate_matrix();
  }
}

int main(int argc, char *argv[]) {

  unsigned num_threads{1};

  unsigned num_rounds{1};

  std::string model = "omp";

  if (argc > 1)
    num_threads = atoi(argv[1]);

  std::cout << "model=" << model << ' ' << "num_threads=" << num_threads << ' '
            << "num_rounds=" << num_rounds << ' ' << std::endl;

  matrix_multiplication(model, num_threads, num_rounds);

  return 0;
}

// matrix_multiplication_omp
// reference: https://computing.llnl.gov/tutorials/openMP/samples/C/omp_mm.c
void matrix_multiplication_omp(unsigned nthreads) {

  omp_set_num_threads(nthreads);

  int i, j, k;

#pragma omp parallel for private(i, j)
  for (i = 0; i < N; ++i) {
    for (j = 0; j < N; j++) {
      a[i][j] = i + j;
    }
  }

#pragma omp parallel for private(i, j)
  for (i = 0; i < N; ++i) {
    for (j = 0; j < N; j++) {
      b[i][j] = i * j;
    }
  }

#pragma omp parallel for private(i, j)
  for (i = 0; i < N; ++i) {
    for (j = 0; j < N; j++) {
      c[i][j] = 0;
    }
  }

#pragma omp parallel for private(i, j, k)
  for (i = 0; i < N; ++i) {
    for (j = 0; j < N; j++) {
      for (k = 0; k < N; k++) {
        c[i][j] += a[i][k] * b[k][j];
      }
    }
  }

  // int edge;

  //#pragma omp parallel shared(a, b, c, nthreads) private(i, j, k)
  //{
  //  #pragma omp single private(i, j)
  //  for(i = 0; i<N; i++) {
  //    #pragma omp task private(j) firstprivate(i) depend(out: edge)
  //    for (j=0; j<N; j++)
  //      a[i][j]= i+j;
  //  }

  //  #pragma omp single private(i, j)
  //  for(i = 0; i<N; i++) {
  //    #pragma omp task private(j) firstprivate(i) depend(out: edge)
  //    for (j=0; j<N; j++)
  //      b[i][j]= i*j;
  //  }

  //  #pragma omp single private(i, j)
  //  for(i = 0; i<N; i++) {
  //    #pragma omp task private(j) firstprivate(i) depend(out: edge)
  //    for (j=0; j<N; j++)
  //      c[i][j]= 0;
  //  }

  //  #pragma omp single private(i, j)
  //  for(i = 0; i<N; i++) {
  //    #pragma omp task private(j, k) firstprivate(i) depend(in: edge)
  //    for(j=0; j<N; j++) {
  //      for (k=0; k<N; k++) {
  //        c[i][j] += a[i][k] * b[k][j];
  //      }
  //    }
  //  }
  //}

  // std::cout << reduce_sum() << std::endl;
}

std::chrono::microseconds measure_time_omp(unsigned num_threads) {
  auto beg = std::chrono::high_resolution_clock::now();
  matrix_multiplication_omp(num_threads);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}
