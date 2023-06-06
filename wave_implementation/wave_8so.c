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


int Kernel(struct dataobj *restrict damp_vec, struct dataobj *restrict u_vec, struct dataobj *restrict vp_vec, const float dt, const int time_M, const int time_m, const int x0_blk0_size, const int x_M, const int x_m, const int y0_blk0_size, const int y_M, const int y_m, const int z_M, const int z_m, const int nthreads, struct profiler * timers)
{
  float (*restrict damp)[damp_vec->size[1]][damp_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[damp_vec->size[1]][damp_vec->size[2]]) damp_vec->data;
  float (*restrict u)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]]) u_vec->data;
  float (*restrict vp)[vp_vec->size[1]][vp_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[vp_vec->size[1]][vp_vec->size[2]]) vp_vec->data;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  float r0 = 1.0F/(dt*dt);
  float r1 = 1.0F/dt;

  int wf_height = atoi(getenv("WF_HEIGHT"));
  int wf_x_width = atoi(getenv("WF_X_WIDTH"));
  int wf_y_width = atoi(getenv("WF_Y_WIDTH"));
  int angle = 4;

  START_TIMER(section0)
  for (int wf_time_base = time_m; wf_time_base <= time_M; wf_time_base += wf_height)
  {
    for (int wf_x_base = x_m; wf_x_base <= x_M + angle * (wf_height - 1); wf_x_base += wf_x_width) 
    {
      for (int wf_y_base = y_m; wf_y_base <= y_M + angle * (wf_height - 1); wf_y_base += wf_y_width) 
      {
        int wf_offset = 0;
        for (int time = wf_time_base, t0 = (time)%(3), t1 = (time + 2)%(3), t2 = (time + 1)%(3); time <= MIN(time_M, wf_time_base + wf_height - 1); time += 1, t0 = (time)%(3), t1 = (time + 2)%(3), t2 = (time + 1)%(3))
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
                    #pragma omp simd aligned(damp,u,vp:32)  
                    for (int z = z_m; z <= z_M; z += 1)
                    {
                      float r2 = 1.0F/(vp[x + 8][y + 8][z + 8]*vp[x + 8][y + 8][z + 8]);
                      u[t2][x + 8][y + 8][z + 8] = (r1*damp[x + 8][y + 8][z + 8]*u[t0][x + 8][y + 8][z + 8] + r2*(-r0*(-2.0F*u[t0][x + 8][y + 8][z + 8]) - r0*u[t1][x + 8][y + 8][z + 8]) + 1.78571425e-5F*(-u[t0][x + 4][y + 8][z + 8] - u[t0][x + 8][y + 4][z + 8] - u[t0][x + 8][y + 8][z + 4] - u[t0][x + 8][y + 8][z + 12] - u[t0][x + 8][y + 12][z + 8] - u[t0][x + 12][y + 8][z + 8]) + 2.53968248e-4F*(u[t0][x + 5][y + 8][z + 8] + u[t0][x + 8][y + 5][z + 8] + u[t0][x + 8][y + 8][z + 5] + u[t0][x + 8][y + 8][z + 11] + u[t0][x + 8][y + 11][z + 8] + u[t0][x + 11][y + 8][z + 8]) + 1.99999996e-3F*(-u[t0][x + 6][y + 8][z + 8] - u[t0][x + 8][y + 6][z + 8] - u[t0][x + 8][y + 8][z + 6] - u[t0][x + 8][y + 8][z + 10] - u[t0][x + 8][y + 10][z + 8] - u[t0][x + 10][y + 8][z + 8]) + 1.59999996e-2F*(u[t0][x + 7][y + 8][z + 8] + u[t0][x + 8][y + 7][z + 8] + u[t0][x + 8][y + 8][z + 7] + u[t0][x + 8][y + 8][z + 9] + u[t0][x + 8][y + 9][z + 8] + u[t0][x + 9][y + 8][z + 8]) - 8.54166647e-2F*u[t0][x + 8][y + 8][z + 8])/(r0*r2 + r1*damp[x + 8][y + 8][z + 8]);
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
