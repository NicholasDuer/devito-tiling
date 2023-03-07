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
#include "mpi.h"

struct dataobj
{
  void *restrict data;
  unsigned long * size;
  unsigned long * nbsize;
  unsigned long * dsize;
  int * hsize;
  int * hofs;
  int * oofs;
  void * dmap;
} ;

struct neighborhood
{
  int lll, llc, llr, lcl, lcc, lcr, lrl, lrc, lrr;
  int cll, clc, clr, ccl, ccc, ccr, crl, crc, crr;
  int rll, rlc, rlr, rcl, rcc, rcr, rrl, rrc, rrr;
} ;

struct profiler
{
  double section0;
} ;

static void gather0(float *restrict buf_vec, const int x_size, const int y_size, const int z_size, struct dataobj *restrict u_vec, const int otime, const int ox, const int oy, const int oz);
static void scatter0(float *restrict buf_vec, const int x_size, const int y_size, const int z_size, struct dataobj *restrict u_vec, const int otime, const int ox, const int oy, const int oz);
static void sendrecv0(struct dataobj *restrict u_vec, const int x_size, const int y_size, const int z_size, int ogtime, int ogx, int ogy, int ogz, int ostime, int osx, int osy, int osz, int fromrank, int torank, MPI_Comm comm);
static void haloupdate0(struct dataobj *restrict u_vec, MPI_Comm comm, struct neighborhood * nb, int otime);

int angle = 1;
int time_tile_size = 3;

static int checkisleft(struct neighborhood * nb) {
  if (nb->lll == MPI_PROC_NULL && nb->llc == MPI_PROC_NULL && nb->llr == MPI_PROC_NULL && nb->lcl == MPI_PROC_NULL && nb->lcc == MPI_PROC_NULL && nb->lcr == MPI_PROC_NULL && nb->lrl == MPI_PROC_NULL && nb->lrc == MPI_PROC_NULL && nb->lrr == MPI_PROC_NULL) {
    return 1;
  }
  return 0;
}

static int checkisright(struct neighborhood * nb) {
  if (nb->rll == MPI_PROC_NULL && nb->rlc == MPI_PROC_NULL && nb->rlr == MPI_PROC_NULL && nb->rcl == MPI_PROC_NULL && nb->rcc == MPI_PROC_NULL && nb->rcr == MPI_PROC_NULL && nb->rrl == MPI_PROC_NULL && nb->rrc == MPI_PROC_NULL && nb->rrr == MPI_PROC_NULL) {
    return 1;
  }
  return 0;
}

static int checkistop(struct neighborhood * nb) {
  if (nb->lrl == MPI_PROC_NULL && nb->lrc == MPI_PROC_NULL && nb->lrr == MPI_PROC_NULL && nb->crl == MPI_PROC_NULL && nb->crc == MPI_PROC_NULL && nb->crr == MPI_PROC_NULL && nb->rrl == MPI_PROC_NULL && nb->rrc == MPI_PROC_NULL && nb->rrr == MPI_PROC_NULL) {
    return 1;
  }
  return 0;
}

static int checkisbottom(struct neighborhood * nb) {
  if (nb->lll == MPI_PROC_NULL && nb->llc == MPI_PROC_NULL && nb->llr == MPI_PROC_NULL && nb->cll == MPI_PROC_NULL && nb->clc == MPI_PROC_NULL && nb->clr == MPI_PROC_NULL && nb->rll == MPI_PROC_NULL && nb->rlc == MPI_PROC_NULL && nb->rlr == MPI_PROC_NULL) {
    return 1;
  }
  return 0;
}

int Kernel(struct dataobj *restrict u_vec, const float h_x, const float h_y, const float h_z, const int time_M, const int time_m, const int x0_blk0_size, const int x_M, const int x_m, const int y0_blk0_size, const int y_M, const int y_m, const int z_M, const int z_m, MPI_Comm comm, struct neighborhood * nb, struct profiler * timers)
{
  float (*restrict u)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]]) u_vec->data;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  float r0 = 1.0F/h_y;
  float r1 = 1.0F/h_x;
  float r2 = 1.0F/h_z;

  int isleft = checkisleft(nb);
  int isright = checkisright(nb);
  int istop = checkistop(nb);
  int isbottom = checkisbottom(nb);

  for (int time_tile_base = time_m; time_tile_base <= time_M; time_tile_base += time_tile_size)
  {
    haloupdate0(u_vec,comm,nb,time_tile_base % 2);
    for (int time = time_tile_base, t0 = (time)%(2), t1 = (time + 1)%(2); time <= MIN(time_M, time_tile_base + time_tile_size - 1); time += 1, t0 = (time)%(2), t1 = (time + 1)%(2))
    {
      /* Begin section0 */
      START_TIMER(section0)

      int offset = ((time_tile_size - 1) - (time % time_tile_size)) * angle;
      int lower_x_offset = offset;
      int upper_x_offset = offset; 
      int lower_y_offset = offset; 
      int upper_y_offset = offset;

      if (isleft) {
        lower_x_offset = 0;
      }

      if (isright) {
        upper_x_offset = 0;
      }

      if (isbottom) {
        lower_y_offset = 0;
      }

      if (istop) {
        upper_y_offset = 0;
      }

      for (int x0_blk0 = x_m - lower_x_offset; x0_blk0 <= x_M + upper_x_offset; x0_blk0 += x0_blk0_size)
      {
        for (int y0_blk0 = y_m - lower_y_offset; y0_blk0 <= y_M + upper_y_offset; y0_blk0 += y0_blk0_size)
        {
          for (int x = x0_blk0; x <= MIN(x0_blk0 + x0_blk0_size - 1, x_M + upper_x_offset); x += 1)
          {
            for (int y = y0_blk0; y <= MIN(y0_blk0 + y0_blk0_size - 1, y_M + upper_y_offset); y += 1)
            {
              #pragma omp simd aligned(u:32)
              for (int z = z_m; z <= z_M; z += 1)
              {
                u[t1][x + 3][y + 3][z + 3] = -r0*(-u[t0][x + 3][y + 3][z + 3]) - r0*u[t0][x + 3][y + 4][z + 3] + r1*(-u[t0][x + 3][y + 3][z + 3]) + r1*u[t0][x + 4][y + 3][z + 3] + r2*(-u[t0][x + 3][y + 3][z + 3]) + r2*u[t0][x + 3][y + 3][z + 4] + 1.0e-3F;
              }
            }
          }
        }
      }
      STOP_TIMER(section0,timers)
    /* End section0 */
    }
  }
  return 0;
}

static void gather0(float *restrict buf_vec, const int x_size, const int y_size, const int z_size, struct dataobj *restrict u_vec, const int otime, const int ox, const int oy, const int oz)
{
  float (*restrict buf)[y_size][z_size] __attribute__ ((aligned (64))) = (float (*)[y_size][z_size]) buf_vec;
  float (*restrict u)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]]) u_vec->data;

  const int x_m = 0;
  const int y_m = 0;
  const int z_m = 0;
  const int x_M = x_size - 1;
  const int y_M = y_size - 1;
  const int z_M = z_size - 1;
  for (int x = x_m; x <= x_M; x += 1)
  {
    for (int y = y_m; y <= y_M; y += 1)
    {
      #pragma omp simd aligned(u:32)
      for (int z = z_m; z <= z_M; z += 1)
      {
        buf[x][y][z] = u[otime][x + ox][y + oy][z + oz];
      }
    }
  }
}

static void scatter0(float *restrict buf_vec, const int x_size, const int y_size, const int z_size, struct dataobj *restrict u_vec, const int otime, const int ox, const int oy, const int oz)
{
  float (*restrict buf)[y_size][z_size] __attribute__ ((aligned (64))) = (float (*)[y_size][z_size]) buf_vec;
  float (*restrict u)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]]) u_vec->data;

  const int x_m = 0;
  const int y_m = 0;
  const int z_m = 0;
  const int x_M = x_size - 1;
  const int y_M = y_size - 1;
  const int z_M = z_size - 1;

  for (int x = x_m; x <= x_M; x += 1)
  {
    for (int y = y_m; y <= y_M; y += 1)
    {
      #pragma omp simd aligned(u:32)
      for (int z = z_m; z <= z_M; z += 1)
      {
        u[otime][x + ox][y + oy][z + oz] = buf[x][y][z];
      }
    }
  }
}

static void sendrecv0(struct dataobj *restrict u_vec, const int x_size, const int y_size, const int z_size, int ogtime, int ogx, int ogy, int ogz, int ostime, int osx, int osy, int osz, int fromrank, int torank, MPI_Comm comm)
{
  float *bufg_vec;
  posix_memalign((void**)(&bufg_vec),64,x_size*y_size*z_size*sizeof(float));
  float *bufs_vec;
  posix_memalign((void**)(&bufs_vec),64,x_size*y_size*z_size*sizeof(float));

  MPI_Request rrecv;
  MPI_Request rsend;

  MPI_Irecv(bufs_vec,x_size*y_size*z_size,MPI_FLOAT,fromrank,13,comm,&(rrecv));
  if (torank != MPI_PROC_NULL)
  {
    gather0(bufg_vec,x_size,y_size,z_size,u_vec,ogtime,ogx,ogy,ogz);
  }
  MPI_Isend(bufg_vec,x_size*y_size*z_size,MPI_FLOAT,torank,13,comm,&(rsend));
  MPI_Wait(&(rsend),MPI_STATUS_IGNORE);
  MPI_Wait(&(rrecv),MPI_STATUS_IGNORE);
  if (fromrank != MPI_PROC_NULL)
  {
    scatter0(bufs_vec,x_size,y_size,z_size,u_vec,ostime,osx,osy,osz);
  }

  free(bufg_vec);
  free(bufs_vec);
}

static void haloupdate0(struct dataobj *restrict u_vec, MPI_Comm comm, struct neighborhood * nb, int otime)
{
  sendrecv0(u_vec,u_vec->hsize[3],u_vec->nbsize[2],u_vec->nbsize[3],otime,u_vec->oofs[2],u_vec->hofs[4],u_vec->hofs[6],otime,u_vec->hofs[3],u_vec->hofs[4],u_vec->hofs[6],nb->rcc,nb->lcc,comm);
  sendrecv0(u_vec,u_vec->nbsize[1],u_vec->hsize[5],u_vec->nbsize[3],otime,u_vec->hofs[2],u_vec->oofs[4],u_vec->hofs[6],otime,u_vec->hofs[2],u_vec->hofs[5],u_vec->hofs[6],nb->crc,nb->clc,comm);
  sendrecv0(u_vec,u_vec->nbsize[1],u_vec->nbsize[2],u_vec->hsize[7],otime,u_vec->hofs[2],u_vec->hofs[4],u_vec->oofs[6],otime,u_vec->hofs[2],u_vec->hofs[4],u_vec->hofs[7],nb->ccr,nb->ccl,comm);
}
/* Backdoor edit at Thu Feb 23 17:25:34 2023*/ 
/* Backdoor edit at Thu Feb 23 17:25:49 2023*/ 
/* Backdoor edit at Thu Feb 23 17:26:33 2023*/ 
/* Backdoor edit at Thu Feb 23 17:26:38 2023*/ 
/* Backdoor edit at Thu Feb 23 17:27:38 2023*/ 
/* Backdoor edit at Thu Feb 23 17:28:24 2023*/ 
/* Backdoor edit at Thu Feb 23 17:33:12 2023*/ 
/* Backdoor edit at Thu Feb 23 17:35:56 2023*/ 
/* Backdoor edit at Thu Feb 23 17:36:53 2023*/ 
/* Backdoor edit at Thu Feb 23 17:37:49 2023*/ 
/* Backdoor edit at Thu Feb 23 17:39:39 2023*/ 
