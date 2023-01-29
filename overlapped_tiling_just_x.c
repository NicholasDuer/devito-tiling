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
} ;

struct profiler
{
  double section0;
} ;


int Kernel(const float a, struct dataobj *restrict u_vec, const float dt, const float h_x, const float h_y, const float h_z, const int time_M, const int time_m, const int x0_blk0_size, const int x_M, const int x_m, const int y0_blk0_size, const int y_M, const int y_m, const int z_M, const int z_m, const int nthreads, struct profiler * timers)
{
  float (*restrict u)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]]) u_vec->data;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  float r0 = 1.0F/dt;
  float r1 = 1.0F/(h_x*h_x);
  float r2 = 1.0F/(h_y*h_y);
  float r3 = 1.0F/(h_z*h_z);

  int tile_time_size = 4;
  int tile_x_size = 8;
  int tile_y_size = 8;
  int angle = 1;
  int number_of_values = 0;

  int tile_x_delta = tile_x_size - 2 * angle * (tile_time_size - 1);
  int tile_y_delta = tile_y_size - 2 * angle * (tile_time_size - 1);

  for (int time_tile_base = time_m; time_tile_base <= time_M; time_tile_base += tile_time_size) 
  {
    /* Begin section0 */
    START_TIMER(section0)

    for (int x_tile_base = x_m - tile_x_delta; x_tile_base <= x_M; x_tile_base += tile_x_delta) 
    {
     // for (int y_tile_base = y_m - tile_y_delta; y_tile_base <= y_M; y_tile_base += tile_y_delta) 
    //  { 
        for (int time = time_tile_base, t0 = (time)%(2), t1 = (time + 1)%(2); time <= MIN(time_tile_base + tile_time_size - 1, time_M); time += 1, t0 = (time)%(2), t1 = (time + 1)%(2))
          {
            #pragma omp parallel num_threads(nthreads)
            {
           //   #pragma omp for collapse(2) schedule(dynamic,1)
              #pragma omp for collapse(1) schedule(dynamic,1)

              for (int x0_blk0 = x_tile_base + angle * (time % tile_time_size); x0_blk0 <= MIN(x_M, x_tile_base + tile_x_size - angle * (time % tile_time_size) - 1); x0_blk0 += x0_blk0_size)
              {
            //    for (int y0_blk0 = y_tile_base + angle * (time % tile_time_size); y0_blk0 <= y_M; y0_blk0 += y0_blk0_size)
            //    {
                  for (int x = MAX(x_m, x0_blk0); x <= MIN(MIN(x0_blk0 + x0_blk0_size - 1, x_M), x_tile_base + tile_x_size - angle * (time % tile_time_size) - 1); x += 1)
                  {
      //              for (int y = MAX(y_m, y0_blk0); y <= MIN(MIN(y0_blk0 + y0_blk0_size - 1, y_M), y_tile_base + tile_y_size - angle * (time % tile_time_size) - 1); y += 1)
     //               {
                      if (number_of_values < 4000) {
                        printf("time: %d, x: %d, x0_blk0: %d, x_tile_base: %d, time_tile_base: %d, thread_number: %d\n", time, x, x0_blk0, x_tile_base, time_tile_base, omp_get_thread_num());
                        number_of_values++;
                      }
                      for (int y = y_m; y <= y_M; y++) {

                      #pragma omp simd aligned(u:32)
                      for (int z = z_m; z <= z_M; z += 1)
                      { 
                        float r4 = -2.5F*u[t0][x + 4][y + 4][z + 4];
                        u[t1][x + 4][y + 4][z + 4] = dt*(a*(r1*(r4 - 8.33333333e-2F*(u[t0][x + 2][y + 4][z + 4] + u[t0][x + 6][y + 4][z + 4]) + 1.33333333F*(u[t0][x + 3][y + 4][z + 4] + u[t0][x + 5][y + 4][z + 4])) + r2*(r4 - 8.33333333e-2F*(u[t0][x + 4][y + 2][z + 4] + u[t0][x + 4][y + 6][z + 4]) + 1.33333333F*(u[t0][x + 4][y + 3][z + 4] + u[t0][x + 4][y + 5][z + 4])) + r3*(r4 - 8.33333333e-2F*(u[t0][x + 4][y + 4][z + 2] + u[t0][x + 4][y + 4][z + 6]) + 1.33333333F*(u[t0][x + 4][y + 4][z + 3] + u[t0][x + 4][y + 4][z + 5]))) + r0*u[t0][x + 4][y + 4][z + 4] + 1.0e-1F);

                      }
                      }
   //                 }
                  }
     //           }
              }
            }
            STOP_TIMER(section0,timers)
            /* End section0 */
          }
    //  }
    }
  }

  return 0;
}
/* Backdoor edit at Wed Jan 25 14:13:09 2023*/ 
/* Backdoor edit at Wed Jan 25 14:24:15 2023*/ 
/* Backdoor edit at Wed Jan 25 14:25:06 2023*/ 
/* Backdoor edit at Wed Jan 25 14:25:35 2023*/ 
/* Backdoor edit at Wed Jan 25 14:26:20 2023*/ 
/* Backdoor edit at Wed Jan 25 14:26:44 2023*/ 
/* Backdoor edit at Wed Jan 25 14:29:55 2023*/ 
/* Backdoor edit at Wed Jan 25 14:31:06 2023*/ 
/* Backdoor edit at Wed Jan 25 14:31:27 2023*/ 
/* Backdoor edit at Wed Jan 25 14:31:34 2023*/ 
/* Backdoor edit at Wed Jan 25 14:31:46 2023*/ 
/* Backdoor edit at Wed Jan 25 14:32:18 2023*/ 
/* Backdoor edit at Wed Jan 25 14:32:30 2023*/ 
/* Backdoor edit at Wed Jan 25 14:35:39 2023*/ 
/* Backdoor edit at Wed Jan 25 14:37:06 2023*/ 
/* Backdoor edit at Wed Jan 25 14:38:27 2023*/ 
/* Backdoor edit at Wed Jan 25 14:50:45 2023*/ 
/* Backdoor edit at Wed Jan 25 15:02:43 2023*/ 
/* Backdoor edit at Wed Jan 25 15:03:13 2023*/ 
/* Backdoor edit at Wed Jan 25 15:07:38 2023*/ 
/* Backdoor edit at Wed Jan 25 15:08:35 2023*/ 
/* Backdoor edit at Wed Jan 25 15:19:38 2023*/ 
/* Backdoor edit at Wed Jan 25 15:19:55 2023*/ 
/* Backdoor edit at Wed Jan 25 15:20:21 2023*/ 
/* Backdoor edit at Wed Jan 25 15:21:23 2023*/ 
/* Backdoor edit at Wed Jan 25 15:22:18 2023*/ 
/* Backdoor edit at Wed Jan 25 15:42:49 2023*/ 
/* Backdoor edit at Wed Jan 25 15:42:56 2023*/ 
/* Backdoor edit at Wed Jan 25 15:43:40 2023*/ 
/* Backdoor edit at Wed Jan 25 15:44:39 2023*/ 
/* Backdoor edit at Wed Jan 25 15:45:27 2023*/ 
/* Backdoor edit at Wed Jan 25 16:34:30 2023*/ 
/* Backdoor edit at Wed Jan 25 16:37:00 2023*/ 
/* Backdoor edit at Wed Jan 25 16:39:27 2023*/ 
/* Backdoor edit at Wed Jan 25 16:39:54 2023*/ 
/* Backdoor edit at Wed Jan 25 16:40:31 2023*/ 
/* Backdoor edit at Wed Jan 25 16:41:06 2023*/ 
/* Backdoor edit at Wed Jan 25 16:41:28 2023*/ 
/* Backdoor edit at Wed Jan 25 16:42:14 2023*/ 
/* Backdoor edit at Wed Jan 25 16:42:33 2023*/ 
/* Backdoor edit at Wed Jan 25 16:44:32 2023*/ 
