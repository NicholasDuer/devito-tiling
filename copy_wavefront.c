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


int Kernel(const float a, struct dataobj *restrict u_vec, const float dt, const float h_x, const float h_y, const float h_z, const int time_M, const int time_m, int x0_blk0_size, const int x_M, const int x_m, int y0_blk0_size, const int y_M, const int y_m, const int z_M, const int z_m, struct profiler * timers)
{
  float (*restrict u)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]]) u_vec->data;
  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  float r0 = 1.0F/dt;
  float r1 = 1.0F/(h_x*h_x);
  float r2 = 1.0F/(h_y*h_y);
  float r3 = 1.0F/(h_z*h_z);


  int block_size = 4;
  x0_blk0_size = block_size;
  y0_blk0_size = block_size;
  printf("x_M: %d, y_M: %d, z_M: %d, t_M: %d\n", x_M, y_M, z_M, time_M);
  printf("x0_blk0_size: %d, y0_blk0_size: %d\n", x0_blk0_size, y0_blk0_size);
  
  int visited_points[41][11][11][11] = {0};

  int a_x = 2;
  int a_y = 2;
  int a_z = 0;

  int w_t = 3;  
  int w_x = 2 * x0_blk0_size;
  int w_y = 2 * y0_blk0_size;
  int w_z = z_M;



  for (int t_2 = time_m; t_2 <= time_M; t_2 += w_t) {
    for (int i_2 = x_m; i_2 <= x_M + (w_t * a_x); i_2 += w_x) {
      for (int j_2 = y_m; j_2 <= y_M + (w_t * a_y); j_2 += w_y) {
        for (int k_2 = z_m; k_2 <= z_M + (w_t + a_z); k_2 += w_z) {

          int o_x = 0;
          int o_y = 0;
          int o_z = 0;
          for (int t = t_2, t0 = (t)%(2), t1 = (t + 1)%(2); t <= MIN(t_2 + w_t - 1, time_M); t++, t0 = (t)%(2), t1 = (t + 1)%(2)) {
            for (int i_1 = MAX(0, i_2 - o_x); i_1 <= i_2 - o_x + w_x - 1; i_1 += x0_blk0_size) {
              for (int j_1 = MAX(0, j_2 - o_y); j_1 <= j_2 - o_y + w_y - 1; j_1 += y0_blk0_size) {
                for (int k_1 = MAX(0, k_2 - o_z); k_1 <= k_2 - o_z + w_z - 1; k_1 += z_M) {

                  /* Note here I have 3 upper bounds. Not sure if they are all necessary or not*/
                  for (int i = i_1; i <= MIN(i_2 - o_x + w_x - 1, MIN(i_1 + x0_blk0_size - 1, x_M)); i++) {
                    for (int j = j_1; j <= MIN(j_2 - o_y + w_y - 1, MIN(j_1 + y0_blk0_size - 1, y_M)); j++) {
                      for (int k = k_1; k <= MIN(k_1 + z_M - 1, z_M); k++) {
                              visited_points[t][i][j][k] = visited_points[t][i][j][k] + 1;
                              float r4 = -2.5F*u[t0][i + 4][j + 4][k + 4];
                              u[t1][i + 4][j + 4][k + 4] = dt*(a*(r1*(r4 - 8.33333333e-2F*(u[t0][i + 2][j + 4][k + 4] + u[t0][i + 6][j + 4][k + 4]) + 1.33333333F*(u[t0][i + 3][j + 4][k + 4] + u[t0][i + 5][j + 4][k + 4])) + r2*(r4 - 8.33333333e-2F*(u[t0][i + 4][j + 2][k + 4] + u[t0][i + 4][j + 6][k + 4]) + 1.33333333F*(u[t0][i + 4][j + 3][k + 4] + u[t0][i + 4][j + 5][k + 4])) + r3*(r4 - 8.33333333e-2F*(u[t0][i + 4][j + 4][k + 2] + u[t0][i + 4][j + 4][k + 6]) + 1.33333333F*(u[t0][i + 4][j + 4][k + 3] + u[t0][i + 4][j + 4][k + 5]))) + r0*u[t0][i + 4][j + 4][k + 4] + 1.0e-1F);
                          }
                  }
                }
              }
            }
          }
          o_x += a_x;
          o_y += a_y;
          o_z += a_z;
        }
      }
    }
  }

 
  /*
  for (int time = time_m, t0 = (time)%(2), t1 = (time + 1)%(2); time <= time_M; time += 1, t0 = (time)%(2), t1 = (time + 1)%(2))
  {
   // printf("time: %d, time_M: %d, t0: %d, t1: %d\n", time, time_M, t0, t1);
    START_TIMER(section0)
    for (int x0_blk0 = x_m; x0_blk0 <= x_M; x0_blk0 += x0_blk0_size)
    {
      for (int y0_blk0 = y_m; y0_blk0 <= y_M; y0_blk0 += y0_blk0_size)
      {
        for (int x = x0_blk0; x <= MIN(x0_blk0 + x0_blk0_size - 1, x_M); x += 1)
        {
          for (int y = y0_blk0; y <= MIN(y0_blk0 + y0_blk0_size - 1, y_M); y += 1)
          {
            #pragma omp simd aligned(u:32)
            for (int z = z_m; z <= z_M; z += 1)
            {
    //          printf("x: %d, y: %d, z: %d\n", x, y, z);
              float r4 = -2.5F*u[t0][x + 4][y + 4][z + 4];
              u[t1][x + 4][y + 4][z + 4] = dt*(a*(r1*(r4 - 8.33333333e-2F*(u[t0][x + 2][y + 4][z + 4] + u[t0][x + 6][y + 4][z + 4]) + 1.33333333F*(u[t0][x + 3][y + 4][z + 4] + u[t0][x + 5][y + 4][z + 4])) + r2*(r4 - 8.33333333e-2F*(u[t0][x + 4][y + 2][z + 4] + u[t0][x + 4][y + 6][z + 4]) + 1.33333333F*(u[t0][x + 4][y + 3][z + 4] + u[t0][x + 4][y + 5][z + 4])) + r3*(r4 - 8.33333333e-2F*(u[t0][x + 4][y + 4][z + 2] + u[t0][x + 4][y + 4][z + 6]) + 1.33333333F*(u[t0][x + 4][y + 4][z + 3] + u[t0][x + 4][y + 4][z + 5]))) + r0*u[t0][x + 4][y + 4][z + 4] + 1.0e-1F);
            }
          }
        }
      }
    }
    STOP_TIMER(section0,timers)
  }
*/
  }
  for (int t = 0; t <= time_M; t++) {
    for (int i = 0; i <= x_M; i++) {
      for (int j = 0; j <= y_M; j++) {
        for (int k = 0; k <= z_M; k++) {
          if (visited_points[t][i][j][k] != 1) {
            printf("WRONG POINT: t: %d, x: %d, y: %d, z: %d, VISITED %d times\n", t, i, j, k, visited_points[t][i][j][k]);
          }
        }
      }
    }
  }
  
  return 0;

}
/* Backdoor edit at Tue Dec 20 12:37:46 2022*/ 
/* Backdoor edit at Tue Dec 20 12:37:59 2022*/ 
/* Backdoor edit at Tue Dec 20 12:38:25 2022*/ 
/* Backdoor edit at Tue Dec 20 12:52:20 2022*/ 
/* Backdoor edit at Tue Dec 20 14:14:16 2022*/ 
/* Backdoor edit at Tue Dec 20 14:15:01 2022*/ 
/* Backdoor edit at Tue Dec 20 14:15:30 2022*/ 
/* Backdoor edit at Tue Dec 20 14:15:57 2022*/ 
/* Backdoor edit at Tue Dec 20 14:27:57 2022*/ 
/* Backdoor edit at Tue Dec 20 14:30:23 2022*/ 
/* Backdoor edit at Tue Dec 20 15:49:09 2022*/ 
/* Backdoor edit at Tue Dec 20 15:50:32 2022*/ 
/* Backdoor edit at Tue Dec 20 15:55:24 2022*/ 
/* Backdoor edit at Tue Dec 20 16:07:22 2022*/ 
/* Backdoor edit at Tue Dec 20 16:07:37 2022*/ 
/* Backdoor edit at Tue Dec 20 16:08:05 2022*/ 
/* Backdoor edit at Tue Dec 20 16:10:02 2022*/ 
/* Backdoor edit at Tue Dec 20 16:13:11 2022*/ 
/* Backdoor edit at Tue Dec 20 16:27:48 2022*/ 
/* Backdoor edit at Tue Dec 20 16:29:01 2022*/ 
/* Backdoor edit at Tue Dec 20 16:29:08 2022*/ 
/* Backdoor edit at Tue Dec 20 16:42:57 2022*/ 
/* Backdoor edit at Tue Dec 20 16:43:03 2022*/ 
/* Backdoor edit at Tue Dec 20 16:47:57 2022*/ 
/* Backdoor edit at Tue Dec 20 16:48:05 2022*/ 
/* Backdoor edit at Tue Dec 20 16:55:26 2022*/ 
/* Backdoor edit at Tue Dec 20 16:58:33 2022*/ 
/* Backdoor edit at Tue Dec 20 17:02:41 2022*/ 
/* Backdoor edit at Wed Dec 21 10:44:41 2022*/ 
/* Backdoor edit at Wed Dec 21 11:07:54 2022*/ 
/* Backdoor edit at Wed Dec 21 11:08:25 2022*/ 
/* Backdoor edit at Wed Dec 21 11:08:35 2022*/ 
/* Backdoor edit at Wed Dec 21 11:08:53 2022*/ 
/* Backdoor edit at Wed Dec 21 11:13:53 2022*/ 
/* Backdoor edit at Wed Dec 21 11:15:35 2022*/ 
/* Backdoor edit at Wed Dec 21 11:15:45 2022*/ 
/* Backdoor edit at Wed Dec 21 11:23:13 2022*/ 
/* Backdoor edit at Wed Dec 21 11:34:10 2022*/ 
/* Backdoor edit at Wed Dec 21 11:34:37 2022*/ 
/* Backdoor edit at Wed Dec 21 11:34:48 2022*/ 
/* Backdoor edit at Wed Dec 21 11:35:53 2022*/ 
/* Backdoor edit at Wed Dec 21 11:36:03 2022*/ 
/* Backdoor edit at Wed Dec 21 11:54:50 2022*/ 
/* Backdoor edit at Wed Dec 21 12:02:20 2022*/ 
/* Backdoor edit at Wed Dec 21 12:03:48 2022*/ 
/* Backdoor edit at Wed Dec 21 12:03:53 2022*/ 
/* Backdoor edit at Wed Dec 21 12:06:51 2022*/ 
/* Backdoor edit at Wed Dec 21 12:14:17 2022*/ 
/* Backdoor edit at Wed Dec 21 12:23:11 2022*/ 
/* Backdoor edit at Wed Dec 21 12:23:30 2022*/ 
/* Backdoor edit at Wed Dec 21 12:25:18 2022*/ 
/* Backdoor edit at Wed Dec 21 12:25:35 2022*/ 
/* Backdoor edit at Wed Dec 21 13:11:45 2022*/ 
/* Backdoor edit at Wed Dec 21 13:23:05 2022*/ 
/* Backdoor edit at Wed Dec 21 13:23:12 2022*/ 
/* Backdoor edit at Wed Dec 21 13:24:48 2022*/ 
/* Backdoor edit at Wed Dec 21 13:25:02 2022*/ 
/* Backdoor edit at Wed Dec 21 13:27:01 2022*/ 
/* Backdoor edit at Wed Dec 21 13:31:34 2022*/ 
/* Backdoor edit at Wed Dec 21 13:31:47 2022*/ 
/* Backdoor edit at Wed Dec 21 13:32:02 2022*/ 
/* Backdoor edit at Wed Dec 21 13:32:16 2022*/ 
/* Backdoor edit at Wed Dec 21 13:33:15 2022*/ 
/* Backdoor edit at Wed Dec 21 13:33:22 2022*/ 
/* Backdoor edit at Wed Dec 21 13:33:30 2022*/ 
/* Backdoor edit at Wed Dec 21 13:33:37 2022*/ 
/* Backdoor edit at Wed Dec 21 13:34:53 2022*/ 
/* Backdoor edit at Wed Dec 21 13:35:57 2022*/ 
/* Backdoor edit at Wed Dec 21 13:36:22 2022*/ 
/* Backdoor edit at Wed Dec 21 13:36:46 2022*/ 
/* Backdoor edit at Wed Dec 21 13:37:07 2022*/ 
/* Backdoor edit at Wed Dec 21 13:37:42 2022*/ 
/* Backdoor edit at Wed Dec 21 13:38:01 2022*/ 
/* Backdoor edit at Wed Dec 21 13:38:08 2022*/ 
/* Backdoor edit at Wed Dec 21 13:38:46 2022*/ 
/* Backdoor edit at Wed Dec 21 13:39:03 2022*/ 
/* Backdoor edit at Wed Dec 21 13:39:10 2022*/ 
/* Backdoor edit at Wed Dec 21 13:47:43 2022*/ 
/* Backdoor edit at Wed Dec 21 13:48:11 2022*/ 
/* Backdoor edit at Wed Dec 21 13:48:25 2022*/ 
/* Backdoor edit at Wed Dec 21 13:50:16 2022*/ 
/* Backdoor edit at Wed Dec 21 13:50:26 2022*/ 
/* Backdoor edit at Wed Dec 21 13:50:41 2022*/ 
/* Backdoor edit at Wed Dec 21 15:43:31 2022*/ 
/* Backdoor edit at Wed Dec 21 15:44:44 2022*/ 
/* Backdoor edit at Wed Dec 21 15:45:12 2022*/ 
/* Backdoor edit at Wed Dec 21 15:47:55 2022*/ 
/* Backdoor edit at Wed Dec 21 15:48:39 2022*/ 
/* Backdoor edit at Wed Dec 21 15:49:03 2022*/ 
/* Backdoor edit at Wed Dec 21 15:49:27 2022*/ 
/* Backdoor edit at Wed Dec 21 15:49:33 2022*/ 
/* Backdoor edit at Wed Dec 21 16:22:26 2022*/ 
/* Backdoor edit at Mon Dec 26 12:34:24 2022*/ 
/* Backdoor edit at Mon Dec 26 12:35:01 2022*/ 
/* Backdoor edit at Mon Dec 26 12:35:12 2022*/ 
/* Backdoor edit at Mon Dec 26 12:38:48 2022*/ 
/* Backdoor edit at Mon Dec 26 12:39:12 2022*/ 
/* Backdoor edit at Mon Dec 26 12:41:40 2022*/ 
/* Backdoor edit at Mon Dec 26 12:41:49 2022*/ 
/* Backdoor edit at Mon Dec 26 12:42:35 2022*/ 
/* Backdoor edit at Mon Dec 26 13:04:58 2022*/ 
/* Backdoor edit at Mon Dec 26 13:06:05 2022*/ 
/* Backdoor edit at Mon Dec 26 13:08:44 2022*/ 
/* Backdoor edit at Mon Dec 26 18:25:50 2022*/ 
/* Backdoor edit at Mon Dec 26 18:28:42 2022*/ 
/* Backdoor edit at Mon Dec 26 18:34:55 2022*/ 
/* Backdoor edit at Mon Dec 26 18:36:07 2022*/ 
/* Backdoor edit at Mon Dec 26 18:36:28 2022*/ 
/* Backdoor edit at Mon Dec 26 18:37:33 2022*/ 
/* Backdoor edit at Mon Dec 26 18:38:24 2022*/ 
/* Backdoor edit at Mon Dec 26 18:38:44 2022*/ 
/* Backdoor edit at Mon Dec 26 18:38:59 2022*/ 
/* Backdoor edit at Mon Dec 26 18:43:54 2022*/ 
/* Backdoor edit at Mon Dec 26 18:48:20 2022*/ 
/* Backdoor edit at Mon Dec 26 18:49:03 2022*/ 
/* Backdoor edit at Thu Dec 29 12:23:28 2022*/ 
/* Backdoor edit at Thu Dec 29 12:23:39 2022*/ 
