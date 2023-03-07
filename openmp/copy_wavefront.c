#define _POSIX_C_SOURCE 200809L
#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define START_TIMER(S) struct timeval start_ ## S , end_ ## S ; gettimeofday(&start_ ## S , NULL);
#define STOP_TIMER(S,T) gettimeofday(&end_ ## S, NULL); T->S += (double)(end_ ## S .tv_sec-start_ ## S.tv_sec)+(double)(end_ ## S .tv_usec-start_ ## S .tv_usec)/1000000;

#include "omp.h"
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


int Kernel(const float a, struct dataobj *restrict u_vec, const float dt, const float h_x, const float h_y, const float h_z, const int time_M, const int time_m, int x0_blk0_size, const int x_M, const int x_m, int y0_blk0_size, const int y_M, const int y_m, const int z_M, const int z_m, int nthreads, struct profiler * timers)
{
  float (*restrict u)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]]) u_vec->data;
  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);


  float r0 = 1.0F/dt;
  float r1 = 1.0F/(h_x*h_x);
  float r2 = 1.0F/(h_y*h_y);
  float r3 = 1.0F/(h_z*h_z);

  x0_blk0_size = 64;
  y0_blk0_size = 64;
  nthreads = 4;

  printf("x_M: %d, y_M: %d, z_M: %d, t_M: %d\n", x_M, y_M, z_M, time_M);
  printf("x0_blk0_size: %d, y0_blk0_size: %d\n", x0_blk0_size, y0_blk0_size);

  
  //int visited_points[41][11][11][11] = {0};

  int a_x = 2;
  int a_y = 2;
  int a_z = 0;
  int wavefront = 0;

  int w_t = 2;  
  int w_x = 2 * x0_blk0_size;
  int w_y = 2 * y0_blk0_size;
  int w_z = z_M;


  if (wavefront) {
  for (int t_2 = time_m; t_2 <= time_M; t_2 += w_t) {
    START_TIMER(section0)
    #pragma omp parallel num_threads( nthreads )
    for (int x_2 = x_m; x_2 <= x_M + (w_t * a_x); x_2 += w_x) {

      for (int y_2 = y_m; y_2 <= y_M + (w_t * a_y); y_2 += w_y) {
        for (int z_2 = z_m; z_2 <= z_M + (w_t + a_z); z_2 += w_z) {

          int o_x = 0;
          int o_y = 0;
          int o_z = 0;
          for (int t = t_2, t0 = (t)%(2), t1 = (t + 1)%(2); t <= MIN(t_2 + w_t - 1, time_M); t++, t0 = (t)%(2), t1 = (t + 1)%(2)) {
            for (int x_1 = MAX(0, x_2 - o_x); x_1 <= x_2 - o_x + w_x - 1; x_1 += x0_blk0_size) {
              for (int y_1 = MAX(0, y_2 - o_y); y_1 <= y_2 - o_y + w_y - 1; y_1 += y0_blk0_size) {
                for (int z_1 = MAX(0, z_2 - o_z); z_1 <= z_2 - o_z + w_z - 1; z_1 += z_M) {

                  /* Note here I have 3 upper bounds. Not sure if they are all necessary or not*/
                  for (int x = x_1; x <= MIN(x_2 - o_x + w_x - 1, MIN(x_1 + x0_blk0_size - 1, x_M)); x++) {
                    for (int y = y_1; y <= MIN(y_2 - o_y + w_y - 1, MIN(y_1 + y0_blk0_size - 1, y_M)); y++) {
                      #pragma omp simd aligned(u:32)

                      for (int z = z_1; z <= MIN(z_1 + z_M - 1, z_M); z++) {
                              //visited_points[t][x][y][z] = visited_points[t][x][y][z] + 1;

                              float r4 = -2.5F*u[t0][x + 4][y + 4][z + 4];
                              u[t1][x + 4][y + 4][z + 4] = dt*(a*(r1*(r4 - 8.33333333e-2F*(u[t0][x + 2][y + 4][z + 4] + u[t0][x + 6][y + 4][z + 4]) + 1.33333333F*(u[t0][x + 3][y + 4][z + 4] + u[t0][x + 5][y + 4][z + 4])) + r2*(r4 - 8.33333333e-2F*(u[t0][x + 4][y + 2][z + 4] + u[t0][x + 4][y + 6][z + 4]) + 1.33333333F*(u[t0][x + 4][y + 3][z + 4] + u[t0][x + 4][y + 5][z + 4])) + r3*(r4 - 8.33333333e-2F*(u[t0][x + 4][y + 4][z + 2] + u[t0][x + 4][y + 4][z + 6]) + 1.33333333F*(u[t0][x + 4][y + 4][z + 3] + u[t0][x + 4][y + 4][z + 5]))) + r0*u[t0][x + 4][y + 4][z + 4] + 1.0e-1F);
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
    STOP_TIMER(section0,timers);
  }
  }
  else {
 
  for (int time = time_m, t0 = (time)%(2), t1 = (time + 1)%(2); time <= time_M; time += 1, t0 = (time)%(2), t1 = (time + 1)%(2))
  {
   // printf("time: %d, time_M: %d, t0: %d, t1: %d\n", time, time_M, t0, t1);
    START_TIMER(section0)
    #pragma omp parallel num_threads( nthreads )

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
  }

/*
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
  */
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
/* Backdoor edit at Sat Jan  7 13:56:41 2023*/ 
/* Backdoor edit at Sat Jan  7 14:05:29 2023*/ 
/* Backdoor edit at Sat Jan  7 14:15:35 2023*/ 
/* Backdoor edit at Sat Jan  7 14:16:07 2023*/ 
/* Backdoor edit at Sat Jan  7 14:16:21 2023*/ 
/* Backdoor edit at Sat Jan  7 14:16:32 2023*/ 
/* Backdoor edit at Sat Jan  7 14:17:55 2023*/ 
/* Backdoor edit at Sat Jan  7 14:19:09 2023*/ 
/* Backdoor edit at Sat Jan  7 14:20:49 2023*/ 
/* Backdoor edit at Sat Jan  7 14:21:06 2023*/ 
/* Backdoor edit at Sat Jan  7 14:22:07 2023*/ 
/* Backdoor edit at Sat Jan  7 14:23:02 2023*/ 
/* Backdoor edit at Sat Jan  7 14:23:35 2023*/ 
/* Backdoor edit at Sat Jan  7 14:24:00 2023*/ 
/* Backdoor edit at Sat Jan  7 14:24:26 2023*/ 
/* Backdoor edit at Sat Jan  7 14:25:16 2023*/ 
/* Backdoor edit at Sat Jan  7 14:25:46 2023*/ 
/* Backdoor edit at Sat Jan  7 14:26:19 2023*/ 
/* Backdoor edit at Sat Jan  7 14:27:04 2023*/ 
/* Backdoor edit at Sat Jan  7 14:30:03 2023*/ 
/* Backdoor edit at Sat Jan  7 14:58:43 2023*/ 
/* Backdoor edit at Sat Jan  7 15:00:47 2023*/ 
/* Backdoor edit at Sat Jan  7 15:01:17 2023*/ 
/* Backdoor edit at Sun Jan  8 17:39:32 2023*/ 
/* Backdoor edit at Sun Jan  8 17:40:30 2023*/ 
/* Backdoor edit at Sun Jan  8 17:41:07 2023*/ 
/* Backdoor edit at Sun Jan  8 17:42:28 2023*/ 
/* Backdoor edit at Sun Jan  8 17:43:32 2023*/ 
/* Backdoor edit at Sun Jan  8 17:44:06 2023*/ 
/* Backdoor edit at Sun Jan  8 17:44:57 2023*/ 
/* Backdoor edit at Sun Jan  8 17:45:42 2023*/ 
/* Backdoor edit at Sun Jan  8 17:46:17 2023*/ 
/* Backdoor edit at Sun Jan  8 17:47:01 2023*/ 
/* Backdoor edit at Sun Jan  8 17:48:01 2023*/ 
/* Backdoor edit at Sun Jan  8 17:48:39 2023*/ 
/* Backdoor edit at Sun Jan  8 17:49:32 2023*/ 
/* Backdoor edit at Sun Jan  8 17:50:26 2023*/ 
/* Backdoor edit at Sun Jan  8 17:51:04 2023*/ 
/* Backdoor edit at Sun Jan  8 17:51:42 2023*/ 
/* Backdoor edit at Sun Jan  8 17:52:14 2023*/ 
/* Backdoor edit at Sun Jan  8 17:52:48 2023*/ 
/* Backdoor edit at Sun Jan  8 17:53:26 2023*/ 
/* Backdoor edit at Sun Jan  8 17:53:58 2023*/ 
/* Backdoor edit at Sun Jan  8 17:54:50 2023*/ 
/* Backdoor edit at Sun Jan  8 17:55:37 2023*/ 
/* Backdoor edit at Sun Jan  8 17:56:34 2023*/ 
/* Backdoor edit at Sun Jan  8 17:57:55 2023*/ 
/* Backdoor edit at Sun Jan  8 17:58:46 2023*/ 
/* Backdoor edit at Sun Jan  8 17:59:54 2023*/ 
/* Backdoor edit at Sun Jan  8 18:00:42 2023*/ 
/* Backdoor edit at Sun Jan  8 18:05:58 2023*/ 
/* Backdoor edit at Mon Jan  9 13:22:12 2023*/ 
/* Backdoor edit at Mon Jan  9 13:22:56 2023*/ 
/* Backdoor edit at Mon Jan  9 13:26:02 2023*/ 
/* Backdoor edit at Mon Jan  9 13:26:33 2023*/ 
/* Backdoor edit at Mon Jan  9 13:27:38 2023*/ 
/* Backdoor edit at Mon Jan  9 13:52:10 2023*/ 
/* Backdoor edit at Mon Jan  9 13:55:05 2023*/ 
/* Backdoor edit at Mon Jan  9 13:56:12 2023*/ 
/* Backdoor edit at Mon Jan  9 13:56:28 2023*/ 
/* Backdoor edit at Mon Jan  9 13:57:23 2023*/ 
/* Backdoor edit at Mon Jan  9 13:58:01 2023*/ 
/* Backdoor edit at Mon Jan  9 14:29:20 2023*/ 
/* Backdoor edit at Mon Jan  9 14:29:50 2023*/ 
/* Backdoor edit at Mon Jan  9 14:30:36 2023*/ 
/* Backdoor edit at Mon Jan  9 14:31:00 2023*/ 
/* Backdoor edit at Mon Jan  9 14:31:18 2023*/ 
/* Backdoor edit at Mon Jan  9 14:31:27 2023*/ 
/* Backdoor edit at Mon Jan  9 14:33:01 2023*/ 
/* Backdoor edit at Mon Jan  9 14:33:12 2023*/ 
/* Backdoor edit at Mon Jan  9 14:33:24 2023*/ 
/* Backdoor edit at Mon Jan  9 14:33:36 2023*/ 
/* Backdoor edit at Mon Jan  9 14:47:52 2023*/ 
/* Backdoor edit at Mon Jan  9 14:51:16 2023*/ 
/* Backdoor edit at Mon Jan  9 15:00:55 2023*/ 
/* Backdoor edit at Mon Jan  9 15:01:35 2023*/ 
/* Backdoor edit at Mon Jan  9 15:01:47 2023*/ 
/* Backdoor edit at Mon Jan  9 15:03:53 2023*/ 
/* Backdoor edit at Mon Jan  9 15:08:34 2023*/ 
/* Backdoor edit at Mon Jan  9 15:09:09 2023*/ 
/* Backdoor edit at Mon Jan  9 15:09:46 2023*/ 
/* Backdoor edit at Mon Jan  9 15:10:03 2023*/ 
/* Backdoor edit at Mon Jan  9 15:10:52 2023*/ 
/* Backdoor edit at Mon Jan  9 15:11:36 2023*/ 
/* Backdoor edit at Mon Jan  9 15:12:08 2023*/ 
/* Backdoor edit at Mon Jan  9 15:12:41 2023*/ 
/* Backdoor edit at Mon Jan  9 15:14:06 2023*/ 
/* Backdoor edit at Mon Jan  9 15:14:58 2023*/ 
/* Backdoor edit at Mon Jan  9 15:15:29 2023*/ 
/* Backdoor edit at Mon Jan  9 15:43:10 2023*/ 
/* Backdoor edit at Mon Jan  9 15:43:44 2023*/ 
/* Backdoor edit at Mon Jan  9 15:44:15 2023*/ 
/* Backdoor edit at Mon Jan  9 15:44:50 2023*/ 
/* Backdoor edit at Mon Jan  9 15:45:53 2023*/ 
/* Backdoor edit at Mon Jan  9 15:46:35 2023*/ 
/* Backdoor edit at Mon Jan  9 15:47:05 2023*/ 
/* Backdoor edit at Mon Jan  9 15:47:48 2023*/ 
/* Backdoor edit at Mon Jan  9 15:48:48 2023*/ 
/* Backdoor edit at Mon Jan  9 15:50:54 2023*/ 
/* Backdoor edit at Mon Jan  9 15:52:23 2023*/ 
/* Backdoor edit at Mon Jan  9 15:53:00 2023*/ 
/* Backdoor edit at Mon Jan  9 15:53:38 2023*/ 
/* Backdoor edit at Mon Jan  9 15:54:13 2023*/ 
/* Backdoor edit at Mon Jan  9 15:55:04 2023*/ 
/* Backdoor edit at Mon Jan  9 15:55:40 2023*/ 
/* Backdoor edit at Mon Jan  9 15:56:16 2023*/ 
/* Backdoor edit at Mon Jan  9 15:57:22 2023*/ 
/* Backdoor edit at Mon Jan  9 15:57:55 2023*/ 
/* Backdoor edit at Mon Jan  9 15:58:44 2023*/ 
/* Backdoor edit at Mon Jan  9 15:59:22 2023*/ 
/* Backdoor edit at Mon Jan  9 16:00:08 2023*/ 
/* Backdoor edit at Mon Jan  9 16:00:42 2023*/ 
/* Backdoor edit at Mon Jan  9 16:01:15 2023*/ 
/* Backdoor edit at Mon Jan  9 16:02:11 2023*/ 
/* Backdoor edit at Mon Jan  9 16:02:42 2023*/ 
/* Backdoor edit at Mon Jan  9 16:03:11 2023*/ 
/* Backdoor edit at Mon Jan  9 16:03:44 2023*/ 
/* Backdoor edit at Mon Jan  9 16:04:33 2023*/ 
/* Backdoor edit at Mon Jan  9 16:04:59 2023*/ 
/* Backdoor edit at Mon Jan  9 16:05:34 2023*/ 
/* Backdoor edit at Mon Jan  9 16:06:03 2023*/ 
/* Backdoor edit at Mon Jan  9 16:07:01 2023*/ 
/* Backdoor edit at Mon Jan  9 16:07:35 2023*/ 
/* Backdoor edit at Mon Jan  9 16:08:07 2023*/ 
/* Backdoor edit at Mon Jan  9 16:08:55 2023*/ 
/* Backdoor edit at Mon Jan  9 16:09:29 2023*/ 
/* Backdoor edit at Mon Jan  9 16:09:59 2023*/ 
/* Backdoor edit at Mon Jan  9 16:11:12 2023*/ 
/* Backdoor edit at Mon Jan  9 16:11:44 2023*/ 
/* Backdoor edit at Mon Jan  9 16:12:22 2023*/ 
/* Backdoor edit at Mon Jan  9 16:13:16 2023*/ 
/* Backdoor edit at Mon Jan  9 16:13:48 2023*/ 
/* Backdoor edit at Mon Jan  9 16:14:18 2023*/ 
/* Backdoor edit at Mon Jan  9 16:15:07 2023*/ 
/* Backdoor edit at Mon Jan  9 16:15:39 2023*/ 
/* Backdoor edit at Mon Jan  9 16:16:14 2023*/ 
/* Backdoor edit at Mon Jan  9 16:17:07 2023*/ 
/* Backdoor edit at Mon Jan  9 16:17:37 2023*/ 
/* Backdoor edit at Mon Jan  9 16:18:08 2023*/ 
/* Backdoor edit at Mon Jan  9 16:18:46 2023*/ 
/* Backdoor edit at Mon Jan  9 16:19:19 2023*/ 
/* Backdoor edit at Mon Jan  9 16:19:53 2023*/ 
/* Backdoor edit at Mon Jan  9 16:20:33 2023*/ 
/* Backdoor edit at Mon Jan  9 16:21:08 2023*/ 
/* Backdoor edit at Mon Jan  9 16:24:06 2023*/ 
/* Backdoor edit at Mon Jan  9 16:24:37 2023*/ 
/* Backdoor edit at Mon Jan  9 16:25:26 2023*/ 
/* Backdoor edit at Mon Jan  9 16:25:59 2023*/ 
/* Backdoor edit at Mon Jan  9 16:26:28 2023*/ 
/* Backdoor edit at Mon Jan  9 16:27:18 2023*/ 
/* Backdoor edit at Mon Jan  9 16:27:50 2023*/ 
/* Backdoor edit at Mon Jan  9 16:28:21 2023*/ 
/* Backdoor edit at Mon Jan  9 16:29:20 2023*/ 
/* Backdoor edit at Mon Jan  9 16:29:47 2023*/ 
/* Backdoor edit at Mon Jan  9 16:30:16 2023*/ 
/* Backdoor edit at Mon Jan  9 16:30:51 2023*/ 
/* Backdoor edit at Mon Jan  9 16:31:19 2023*/ 
/* Backdoor edit at Mon Jan  9 16:31:48 2023*/ 
/* Backdoor edit at Mon Jan  9 16:33:04 2023*/ 
/* Backdoor edit at Mon Jan  9 16:33:38 2023*/ 
/* Backdoor edit at Mon Jan  9 16:34:06 2023*/ 
/* Backdoor edit at Mon Jan  9 16:34:36 2023*/ 
/* Backdoor edit at Mon Jan  9 16:35:35 2023*/ 
/* Backdoor edit at Mon Jan  9 16:36:13 2023*/ 
/* Backdoor edit at Mon Jan  9 16:36:42 2023*/ 
/* Backdoor edit at Mon Jan  9 16:37:14 2023*/ 
/* Backdoor edit at Mon Jan  9 16:38:10 2023*/ 
/* Backdoor edit at Mon Jan  9 16:38:43 2023*/ 
/* Backdoor edit at Mon Jan  9 16:39:20 2023*/ 
/* Backdoor edit at Mon Jan  9 16:39:58 2023*/ 
/* Backdoor edit at Mon Jan  9 16:40:46 2023*/ 
/* Backdoor edit at Mon Jan  9 16:41:12 2023*/ 
/* Backdoor edit at Mon Jan  9 16:41:42 2023*/ 
/* Backdoor edit at Mon Jan  9 16:42:08 2023*/ 
