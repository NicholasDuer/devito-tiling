#define _POSIX_C_SOURCE 200809L
#define START_TIMER(S) struct timeval start_ ## S , end_ ## S ; gettimeofday(&start_ ## S , NULL);
#define STOP_TIMER(S,T) gettimeofday(&end_ ## S, NULL); T->S += (double)(end_ ## S .tv_sec-start_ ## S.tv_sec)+(double)(end_ ## S .tv_usec-start_ ## S .tv_usec)/1000000;
#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#define MAX(a,b) (((a) > (b)) ? (a) : (b))

#include "stdlib.h"
#include "math.h"
#include "sys/time.h"
#include "xmmintrin.h"
#include "pmmintrin.h"
#include "omp.h"

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

const int wf_height = 3;
const int wf_width_x = 4;
const int angle = 1;

int Kernel(struct dataobj *restrict u_vec, const float h_x, const int time_M, const int time_m, const int x_M, const int x_m, const int nthreads, struct profiler * timers)
{
  float (*restrict u)[u_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[u_vec->size[1]]) u_vec->data;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  float r0 = 1.0F/h_x;

  for (int wf_time_base = time_m; wf_time_base <= time_M; wf_time_base += wf_height) 
  { 
    for (int wf_x_base = x_m; wf_x_base <= x_M; wf_x_base += wf_width_x)
    {
      for (int time = wf_time_base, t0 = (time)%(2), t1 = (time + 1)%(2); time <= MIN(time_M, wf_time_base + wf_height - 1); time += 1, t0 = (time)%(2), t1 = (time + 1)%(2))
        {
          int wf_offset = (time % wf_height) * angle;
          START_TIMER(section0)
          {
            for (int x = MAX(x_m, wf_x_base - wf_offset); x <= MIN(x_M, wf_x_base + wf_width_x - wf_offset - 1); x += 1)
            {
	      printf("time: %d, x: %d, wf_x_base: %d\n", time, x, wf_x_base);
              u[t1][x + 1] = r0*(-u[t0][x + 1]) + r0*u[t0][x + 2] + 1;
            }
          }
          STOP_TIMER(section0,timers)
          /* End section0 */
        }
    }
      
  }

  
  return 0;
}
