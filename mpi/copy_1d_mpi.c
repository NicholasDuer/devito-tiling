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
  unsigned long * npsize;
  unsigned long * dsize;
  int * hsize;
  int * hofs;
  int * oofs;
  void * dmap;
} ;

struct neighborhood
{
  int l;
  int c;
  int r;
} ;

struct profiler
{
  double section0;
} ;

static void gather0(float *restrict buf_vec, const int x_size, struct dataobj *restrict u_vec, const int otime, const int ox);
static void scatter0(float *restrict buf_vec, const int x_size, struct dataobj *restrict u_vec, const int otime, const int ox);
static void sendrecv0(struct dataobj *restrict u_vec, const int x_size, int ogtime, int ogx, int ostime, int osx, int fromrank, int torank, MPI_Comm comm);
static void haloupdate0(struct dataobj *restrict u_vec, MPI_Comm comm, struct neighborhood * nb, int otime);

int angle = 1;
int time_tile_size = 3;

int Kernel(struct dataobj *restrict u_vec, const float h_x, const int time_M, const int time_m, const int x_M, const int x_m, MPI_Comm comm, struct neighborhood * nb, struct profiler * timers)
{
  float (*restrict u)[u_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[u_vec->size[1]]) u_vec->data;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  float r0 = 1.0F/h_x;

  int rank;
  MPI_Comm_rank(comm, &rank);

  for (int time_tile_base = time_m; time_tile_base <= time_M; time_tile_base += time_tile_size)
  {
   haloupdate0(u_vec,comm,nb,time_tile_base % 2);

    for (int time = time_tile_base, t0 = (time)%(2), t1 = (time + 1)%(2); time <= MIN(time_M, time_tile_base + time_tile_size - 1); time += 1, t0 = (time)%(2), t1 = (time + 1)%(2))
    {

      START_TIMER(section0)
      /* Begin section0 */
      int offset = ((time_tile_size - 1) - (time % time_tile_size)) * angle;
      int lower_offset;
      int upper_offset;

      if (nb->l == MPI_PROC_NULL) {
        lower_offset = 0;
        upper_offset = offset;
      }

      if (nb->r == MPI_PROC_NULL) {
        lower_offset = offset;
        upper_offset = 0;
      }

      for (int x = x_m - lower_offset; x <= x_M + upper_offset; x += 1)
      {
        u[t1][x + 3] = r0*(-u[t0][x + 3]) + r0*u[t0][x + 4] + 1;
      }
      STOP_TIMER(section0,timers)
      /* End section0 */
    }
  }
}


static void gather0(float *restrict buf_vec, const int x_size, struct dataobj *restrict u_vec, const int otime, const int ox)
{
  float (*restrict buf) __attribute__ ((aligned (64))) = (float (*)) buf_vec;
  float (*restrict u)[u_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[u_vec->size[1]]) u_vec->data;
  const int x_m = 0;
  const int x_M = x_size - 1;
  for (int x = x_m; x <= x_M; x += 1)
  {
    buf[x] = u[otime][x + ox];
  }
}

static void scatter0(float *restrict buf_vec, const int x_size, struct dataobj *restrict u_vec, const int otime, const int ox)
{
  float (*restrict buf) __attribute__ ((aligned (64))) = (float (*)) buf_vec;
  float (*restrict u)[u_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[u_vec->size[1]]) u_vec->data;

  const int x_m = 0;
  const int x_M = x_size - 1;

  for (int x = x_m; x <= x_M; x += 1)
  {
    u[otime][x + ox] = buf[x];
  }
}

static void sendrecv0(struct dataobj *restrict u_vec, const int x_size, int ogtime, int ogx, int ostime, int osx, int fromrank, int torank, MPI_Comm comm)
{
  float *bufg_vec;
  posix_memalign((void**)(&bufg_vec),64,x_size*sizeof(float));
  float *bufs_vec;
  posix_memalign((void**)(&bufs_vec),64,x_size*sizeof(float));

  MPI_Request rrecv;
  MPI_Request rsend;

  printf("x_size: %d, ogx: %d, osx:%d\n", x_size, ogx, osx);

  MPI_Irecv(bufs_vec,x_size,MPI_FLOAT,fromrank,13,comm,&(rrecv));
  if (torank != MPI_PROC_NULL)
  {
    gather0(bufg_vec,x_size,u_vec,ogtime,ogx);
  }
  MPI_Isend(bufg_vec,x_size,MPI_FLOAT,torank,13,comm,&(rsend));
  MPI_Wait(&(rsend),MPI_STATUS_IGNORE);
  MPI_Wait(&(rrecv),MPI_STATUS_IGNORE);
  if (fromrank != MPI_PROC_NULL)
  {
    scatter0(bufs_vec,x_size,u_vec,ostime,osx);
  }

  free(bufg_vec);
  free(bufs_vec);
}

static void haloupdate0(struct dataobj *restrict u_vec, MPI_Comm comm, struct neighborhood * nb, int otime)
{
  sendrecv0(u_vec,u_vec->hsize[3],otime,u_vec->oofs[2],otime,u_vec->hofs[3],nb->r,nb->l,comm);
}/* Backdoor edit at Wed Feb 22 16:10:49 2023*/ 
