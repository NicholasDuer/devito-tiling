#define _POSIX_C_SOURCE 200809L
#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define START_TIMER(S) struct timeval start_ ## S , end_ ## S ; gettimeofday(&start_ ## S , NULL);
#define STOP_TIMER(S,T) gettimeofday(&end_ ## S, NULL); T->S += (double)(end_ ## S .tv_sec-start_ ## S.tv_sec)+(double)(end_ ## S .tv_usec-start_ ## S .tv_usec)/1000000;

#include "stdlib.h"
#include "math.h"
#include "sys/time.h"
#include "xmmintrin.h"
#include "pmmintrin.h"

struct dataobj
{
  void *restrict data;
  unsigned long * size;
  unsigned long * npsize;
  unsigned long * dsize;
  int * hsize;
  int * hofs;
  int * oofs;
  void * dmap;
} ;

struct profiler
{
  double section0;
} ;


int Kernel(struct dataobj *restrict u_vec, const float dt, const float h_x, const float h_y, const float h_z, const int time_M, const int time_m, const int x0_blk0_size, const int x_M, const int x_m, const int y0_blk0_size, const int y_M, const int y_m, const int z_M, const int z_m, const int nthreads, struct profiler * timers)
{
  float (*restrict u)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]]) u_vec->data;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  float r0 = 1.0F/(h_x*h_x);
  float r1 = 1.0F/(h_y*h_y);
  float r2 = 1.0F/(h_z*h_z);
  float r3 = 1.0F/dt;
  int wf_height = atoi(getenv("WF_HEIGHT"));
  int wf_x_width = atoi(getenv("WF_X_WIDTH"));
  int wf_y_width = atoi(getenv("WF_Y_WIDTH"));
  int angle = 1;

  START_TIMER(section0)
  for (int wf_time_base = time_m; wf_time_base <= time_M; wf_time_base += wf_height)
  {
    for (int wf_x_base = x_m; wf_x_base <= x_M + angle * (wf_height - 1); wf_x_base += wf_x_width) 
    {
      for (int wf_y_base = y_m; wf_y_base <= y_M + angle * (wf_height - 1); wf_y_base += wf_y_width) 
      {
        int wf_offset = 0;
        for (int time = wf_time_base, t0 = (time)%(2), t1 = (time + 1)%(2); time <= MIN(time_M, wf_time_base + wf_height - 1); time += 1, t0 = (time)%(2), t1 = (time + 1)%(2))
        {
          #pragma omp parallel num_threads(nthreads)
          {
            #pragma omp for collapse(2) schedule(dynamic,1)
            for (int x0_blk0 = MAX(wf_x_base - wf_offset, x_m); x0_blk0 <= wf_x_base + wf_x_width - wf_offset - 1; x0_blk0 += x0_blk0_size) 
            {
              for (int y0_blk0 = MAX(wf_y_base - wf_offset, y_m); y0_blk0 <= wf_y_base + wf_y_width - wf_offset - 1; y0_blk0 += y0_blk0_size) 
              {
                for (int x = x0_blk0; x <= MIN(MIN(x_M, wf_x_base + wf_x_width - wf_offset - 1), x0_blk0 + x0_blk0_size - 1); x += 1)
                {
                  for (int y = y0_blk0; y <= MIN(MIN(y_M, wf_y_base + wf_y_width - wf_offset - 1), y0_blk0 + y0_blk0_size - 1); y += 1)
                  {
                    #pragma omp simd aligned(u:32)
                    for (int z = z_m; z <= z_M; z += 1)
                    {
                      u[t1][x + 2][y + 2][z + 2] = dt*(-r0*u[t0][x + 2][y + 2][z + 2] - r1*u[t0][x + 2][y + 2][z + 2] - r2*u[t0][x + 2][y + 2][z + 2] + r3*u[t0][x + 2][y + 2][z + 2] + 5.0e-1F*(r0*u[t0][x + 1][y + 2][z + 2] + r0*u[t0][x + 3][y + 2][z + 2] + r1*u[t0][x + 2][y + 1][z + 2] + r1*u[t0][x + 2][y + 3][z + 2] + r2*u[t0][x + 2][y + 2][z + 1] + r2*u[t0][x + 2][y + 2][z + 3]) + 1.0e-1F);
		                }
                  } 
                }
              }
            }
          }
          wf_offset += angle;
        }
      } 
    }
  }
  STOP_TIMER(section0, timers)

  return 0;
}

