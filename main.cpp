/*
 * ring_allreduce.c
 *
 * Ring Reduce-Scatter, All-Gather, and All-Reduce using Linux RDMA Verbs API.
 * Features:
 *  - Eager vs. Rendezvous comparison in the All-Gather phase
 *  - Micro-chunk Pipelining to overlap communication and local reduction
 *  - Registered memory reuse (avoiding ibv_reg_mr inside hot paths)
 *
 * Build:
 *   gcc -std=c11 -O2 -Wall -Wextra -Wpedantic \
 *       -D_POSIX_C_SOURCE=200809L \
 *       ring_allreduce.c -o ring_allreduce -libverbs
 *
 * Example usage:
 *
 * Two processes:
 *   ./ring_allreduce -myindex 01 \
 *       -list mlx-stud-01 mlx-stud-02 \
 *       -iters 1000 -warmup 50
 *
 * Four processes:
 *   ./ring_allreduce -myindex 01 \
 *       -list mlx-stud-01 mlx-stud-02 mlx-stud-03 mlx-stud-04 \
 *       -iters 1000 -warmup 50
 *
 * Run the same command on every participating host,
 * changing -myindex to 01, 02, 03, or 04 as appropriate.
 */

#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE

#include <netinet/tcp.h>
#include <endian.h>
#include <arpa/inet.h>
#include <errno.h>
#include <infiniband/verbs.h>
#include <netdb.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#define PG_DEFAULT_PORT       18515 //TCP port used during connection setup
#define PG_MAX_HOSTS         4 //maximum number of processes
#define PG_MAX_HOST_LEN       128 //maximum hostname length
#define PG_CQ_SIZE           4096 //number of entries in the completion queue.
#define PG_MAX_WR             512 //maximum outstanding work requests.
#define PG_MICRO_CHUNK_SIZE   4096  // Micro-chunk size in elements for
// pipelining

/*
 * Maximum number of distinct Eager receives that may be pre-posted
 * for one collective phase.
 *
 * With the current maximum of four processes and the current message
 * sizes, 64 slots are more than sufficient.
 */
#define PG_EAGER_RECV_SLOTS   64

#define WRID_EAGER_SEND       0x1001ULL
#define WRID_WRITE_SEND       0x1003ULL
#define WRID_EAGER_SLOT_BASE  0x20000000ULL

typedef enum {
    PROTOCOL_EAGER,
    PROTOCOL_RENDEZVOUS
} PROTOCOL_MODE;

typedef enum {
    PG_INT32,
    PG_FLOAT,
    PG_DOUBLE
} DATATYPE;

typedef enum {
    PG_SUM,
    PG_MAX,
    PG_MIN
} OPERATION;

typedef struct {
    int rank;               // the process’s rank
    int size;               // number of processes
    int base_port;          // TCP bootstrap port
    int ib_port;            // physical RDMA port
    int gid_index;          // GID entry for RoCE
    PROTOCOL_MODE protocol; // eager or rendezvous
    char hosts[PG_MAX_HOSTS][PG_MAX_HOST_LEN]; // list of all participating
    // machines
} pg_config_t;

// The network representation of QP connection information
typedef struct __attribute__((packed)) {
    uint16_t lid;    // InfiniBand Local Identifier
    uint32_t qpn;    // Queue Pair Number
    uint32_t psn;    // Packet Sequence Number
    uint8_t gid[16];  // Global Identifier, mainly needed for RoCE
    uint8_t mtu;      // MTU enum value exchanged during QP bootstrap
} wire_qp_info_t;

typedef struct {
    uint16_t lid;
    uint32_t qpn;
    uint32_t psn;
    union ibv_gid gid;
    enum ibv_mtu mtu;
} qp_info_t;

typedef struct __attribute__((packed)) {
    uint64_t address;  // Remote virtual address
    uint32_t rkey;     // Remote key
    uint32_t bytes;    // Registered region size
} wire_mr_info_t;

typedef struct {
    uint64_t address;
    uint32_t rkey;
    uint32_t bytes;
} remote_mr_info_t;

typedef struct {
    uint32_t flags[PG_MAX_HOSTS];
    uint32_t signal_value;
} rendezvous_control_t;

typedef struct {
    pg_config_t cfg;

    struct ibv_device **device_list;
    struct ibv_context *context;       // RDMA context
    struct ibv_pd *pd;                 // protection domain
    struct ibv_cq *cq;                 // completion queue

    struct ibv_qp *qp_next;        // Local -> Next
    struct ibv_qp *qp_prev;        // Prev -> Local

    struct ibv_port_attr port_attr;
    union ibv_gid local_gid;

    // The TCP ring connections
    int socket_to_next;
    int socket_from_prev;

    qp_info_t next_qp_info;
    qp_info_t prev_qp_info;
    size_t eager_slot_size;

    // Cached work-buffer registration used by pg_all_reduce().
    void *work_buffer;
    size_t work_buffer_bytes;
    struct ibv_mr *work_mr;
    remote_mr_info_t next_remote_mr;

    // Persistent receive-ring state for Eager SEND/RECV traffic.
    int next_expected_eager_slot;

    // One-sided completion flags for Rendezvous All-Gather.
    rendezvous_control_t rendezvous_control;
    struct ibv_mr *rendezvous_control_mr;
    remote_mr_info_t next_remote_control_mr;
    uint32_t rendezvous_epoch;

    /*
     * Registered pool used for pre-posted Eager receives.
     * Each receive gets its own slot, so an early incoming message
     * cannot overwrite another message that has not yet been reduced
     * or copied.
     */
    void *eager_recv_pool;
    struct ibv_mr *eager_recv_mr;

    /*
     * Completions that were polled before the current Eager operation
     * was ready to consume them.
     */
    int pending_eager_send_completions;

    int eager_completed_slots[PG_EAGER_RECV_SLOTS];
    int eager_completed_head;
    int eager_completed_tail;
    int eager_completed_count;
} pg_handle_t;

static pg_config_t g_config;

/* Required public API signatures. */
int connect_process_group(char *servername, void **pg_handle);
int pg_all_reduce(void *sendbuf, void *recvbuf, int count,
                  DATATYPE datatype, OPERATION op, void *pg_handle);
int pg_close(void *pg_handle);

/* ------------------------------------------------------------------------- */
/* Helper Functions & Error Handling                                         */
/* ------------------------------------------------------------------------- */

static size_t datatype_size(DATATYPE datatype) {
  switch (datatype) {
    case PG_INT32:  return sizeof(int32_t);
    case PG_FLOAT:  return sizeof(float);
    case PG_DOUBLE: return sizeof(double);
    default:        return 0;
  }
}

static int enable_tcp_nodelay(int socket_fd)
{
  int enabled = 1;

  if (setsockopt(socket_fd,
                 IPPROTO_TCP,
                 TCP_NODELAY,
                 &enabled,
                 sizeof(enabled)) != 0) {
    perror("setsockopt TCP_NODELAY");
    return -1;
  }

  return 0;
}

static int send_all(int socket_fd, const void *buffer, size_t length) {
  const uint8_t *cursor = buffer;
  while (length > 0) {
    ssize_t sent = send(socket_fd, cursor, length, 0);
    if (sent < 0 && errno == EINTR) continue;
    if (sent <= 0) return -1;
    cursor += sent;
    length -= (size_t)sent;
  }
  return 0;
}

static int recv_all(int socket_fd, void *buffer, size_t length) {
  uint8_t *cursor = buffer;
  while (length > 0) {
    ssize_t received = recv(socket_fd, cursor, length, 0);
    if (received < 0 && errno == EINTR) continue;
    if (received <= 0) return -1;
    cursor += received;
    length -= (size_t)received;
  }
  return 0;
}

static int pg_barrier(pg_handle_t *pg)
{
  const uint8_t arrive_token = 0x5a;
  const uint8_t release_token = 0xa5;
  uint8_t received = 0;

  /*
   * Two-phase ring barrier:
   *
   * Phase 1: rank 0 sends an arrival token around the ring.
   * Phase 2: rank 0 sends a release token around the ring.
   */

  if (pg->cfg.rank == 0) {
    if (send_all(pg->socket_to_next,
                 &arrive_token,
                 sizeof(arrive_token)) != 0) {
      return -1;
    }

    if (recv_all(pg->socket_from_prev,
                 &received,
                 sizeof(received)) != 0 ||
        received != arrive_token) {
      return -1;
    }

    if (send_all(pg->socket_to_next,
                 &release_token,
                 sizeof(release_token)) != 0) {
      return -1;
    }

    if (recv_all(pg->socket_from_prev,
                 &received,
                 sizeof(received)) != 0 ||
        received != release_token) {
      return -1;
    }
  }
  else {
    if (recv_all(pg->socket_from_prev,
                 &received,
                 sizeof(received)) != 0 ||
        received != arrive_token) {
      return -1;
    }

    if (send_all(pg->socket_to_next,
                 &received,
                 sizeof(received)) != 0) {
      return -1;
    }

    if (recv_all(pg->socket_from_prev,
                 &received,
                 sizeof(received)) != 0 ||
        received != release_token) {
      return -1;
    }

    if (send_all(pg->socket_to_next,
                 &received,
                 sizeof(received)) != 0) {
      return -1;
    }
  }

  return 0;
}

/* ------------------------------------------------------------------------- */
/* Local Vector Reduction Implementation                                     */
/* ------------------------------------------------------------------------- */

#define REDUCE_TYPED(TYPE) \
    do { \
        TYPE *dest = (TYPE *)inout; \
        const TYPE *src = (const TYPE *)input; \
        for (size_t i = 0; i < elements; ++i) { \
            if (operation == PG_SUM) dest[i] += src[i]; \
            else if (operation == PG_MAX && src[i] > dest[i]) dest[i] = src[i]; \
            else if (operation == PG_MIN && src[i] < dest[i]) dest[i] = src[i]; \
        } \
    } while (0)

static int reduce_local(void *inout, const void *input, size_t elements,
                        DATATYPE datatype, OPERATION operation) {
  switch (datatype) {
    case PG_INT32:  REDUCE_TYPED(int32_t); return 0;
    case PG_FLOAT:  REDUCE_TYPED(float);  return 0;
    case PG_DOUBLE: REDUCE_TYPED(double); return 0;
    default: return -1;
  }
}
#undef REDUCE_TYPED

/* ------------------------------------------------------------------------- */
/* RDMA Low-Level Operations                                                 */
/* ------------------------------------------------------------------------- */

static int wait_for_completions(pg_handle_t *pg, int expected) {
  int completed = 0;
  while (completed < expected) {
    struct ibv_wc wc[8];
    int count = ibv_poll_cq(pg->cq, 8, wc);
    if (count < 0) return -1;
    for (int i = 0; i < count; ++i) {
      if (wc[i].status != IBV_WC_SUCCESS) {
        fprintf(stderr,
                "Rank %d: RDMA completion failed:\n"
                "  wr_id:      %llu\n"
                "  status:     %s\n"
                "  status id:  %d\n"
                "  opcode:     %d\n"
                "  vendor_err: 0x%x\n"
                "  qp_num:     %u\n",
                pg->cfg.rank,
                (unsigned long long)wc[i].wr_id,
                ibv_wc_status_str(wc[i].status),
                wc[i].status,
                wc[i].opcode,
                wc[i].vendor_err,
                wc[i].qp_num);

        return -1;
      }
      ++completed;
    }
  }
  return 0;
}

static int post_eager_receive_slot(
    pg_handle_t *pg,
    int slot_index,
    size_t bytes)
{
  if (!pg ||
      !pg->eager_recv_pool ||
      !pg->eager_recv_mr ||
      slot_index < 0 ||
      slot_index >= PG_EAGER_RECV_SLOTS ||
      bytes == 0 ||
      bytes > pg->eager_slot_size) {

    fprintf(stderr,
            "Rank %d: invalid Eager receive slot request: "
            "slot=%d, bytes=%zu\n",
            pg ? pg->cfg.rank : -1,
            slot_index,
            bytes);

    return -1;
  }

  uint8_t *slot_buffer =
      (uint8_t *)pg->eager_recv_pool +
      (size_t)slot_index * pg->eager_slot_size;

  struct ibv_sge recv_sge = {
      .addr = (uintptr_t)slot_buffer,
      .length = (uint32_t)bytes,
      .lkey = pg->eager_recv_mr->lkey
  };

  struct ibv_recv_wr recv_wr;
  memset(&recv_wr, 0, sizeof(recv_wr));

  recv_wr.wr_id =
      WRID_EAGER_SLOT_BASE +
      (uint64_t)slot_index;

  recv_wr.sg_list = &recv_sge;
  recv_wr.num_sge = 1;
  recv_wr.next = NULL;

  struct ibv_recv_wr *bad_recv_wr = NULL;

  if (ibv_post_recv(
      pg->qp_prev,
      &recv_wr,
      &bad_recv_wr) != 0) {

    fprintf(stderr,
            "Rank %d: failed posting Eager receive slot %d: %s\n",
            pg->cfg.rank,
            slot_index,
            strerror(errno));

    return -1;
  }

  return 0;
}

static int post_eager_send(
    pg_handle_t *pg,
    const void *send_buf,
    struct ibv_mr *send_mr,
    size_t bytes)
{
  if (!pg ||
      !send_buf ||
      !send_mr ||
      bytes == 0 ||
      bytes > UINT32_MAX) {
    return -1;
  }

  struct ibv_sge send_sge = {
      .addr = (uintptr_t)send_buf,
      .length = (uint32_t)bytes,
      .lkey = send_mr->lkey
  };

  struct ibv_send_wr send_wr;
  memset(&send_wr, 0, sizeof(send_wr));

  send_wr.wr_id = WRID_EAGER_SEND;
  send_wr.sg_list = &send_sge;
  send_wr.num_sge = 1;
  send_wr.opcode = IBV_WR_SEND;
  send_wr.send_flags = IBV_SEND_SIGNALED;
  send_wr.next = NULL;

  struct ibv_send_wr *bad_send_wr = NULL;

  if (ibv_post_send(
      pg->qp_next,
      &send_wr,
      &bad_send_wr) != 0) {

    fprintf(stderr,
            "Rank %d: failed posting Eager send: %s\n",
            pg->cfg.rank,
            strerror(errno));

    return -1;
  }

  return 0;
}

static int save_eager_completed_slot(
    pg_handle_t *pg,
    int slot_index)
{
  if (pg->eager_completed_count >= PG_EAGER_RECV_SLOTS) {
    fprintf(stderr,
            "Rank %d: Eager completion queue overflow\n",
            pg->cfg.rank);
    return -1;
  }

  pg->eager_completed_slots[pg->eager_completed_tail] =
      slot_index;

  pg->eager_completed_tail =
      (pg->eager_completed_tail + 1) %
      PG_EAGER_RECV_SLOTS;

  ++pg->eager_completed_count;

  return 0;
}

static int take_saved_eager_completed_slot(
    pg_handle_t *pg,
    int *slot_index)
{
  if (pg->eager_completed_count <= 0) {
    return 0;
  }

  *slot_index =
      pg->eager_completed_slots[pg->eager_completed_head];

  pg->eager_completed_head =
      (pg->eager_completed_head + 1) %
      PG_EAGER_RECV_SLOTS;

  --pg->eager_completed_count;

  return 1;
}

static int wait_for_eager_pair(
    pg_handle_t *pg,
    int *completed_recv_slot)
{
  int have_send = 0;
  int have_recv = 0;
  int recv_slot = -1;

  if (!pg || !completed_recv_slot) {
    return -1;
  }

  /*
   * Consume completions saved by an earlier CQ poll.
   */
  if (pg->pending_eager_send_completions > 0) {
    --pg->pending_eager_send_completions;
    have_send = 1;
  }

  int saved_result =
      take_saved_eager_completed_slot(
          pg,
          &recv_slot);

  if (saved_result < 0) {
    return -1;
  }

  if (saved_result > 0) {
    have_recv = 1;
  }

  while (!have_send || !have_recv) {
    struct ibv_wc wc[8];

    int count =
        ibv_poll_cq(
            pg->cq,
            8,
            wc);

    if (count < 0) {
      fprintf(stderr,
              "Rank %d: ibv_poll_cq failed while waiting "
              "for Eager completions\n",
              pg->cfg.rank);
      return -1;
    }

    for (int i = 0; i < count; ++i) {
      if (wc[i].status != IBV_WC_SUCCESS) {
        fprintf(stderr,
                "Rank %d: Eager RDMA completion failed:\n"
                "  wr_id:      %llu\n"
                "  status:     %s\n"
                "  status id:  %d\n"
                "  opcode:     %d\n"
                "  vendor_err: 0x%x\n"
                "  qp_num:     %u\n",
                pg->cfg.rank,
                (unsigned long long)wc[i].wr_id,
                ibv_wc_status_str(wc[i].status),
                wc[i].status,
                wc[i].opcode,
                wc[i].vendor_err,
                wc[i].qp_num);

        return -1;
      }

      if (wc[i].wr_id == WRID_EAGER_SEND) {
        if (!have_send) {
          have_send = 1;
        }
        else {
          ++pg->pending_eager_send_completions;
        }
      }
      else if (wc[i].wr_id >= WRID_EAGER_SLOT_BASE &&
               wc[i].wr_id <
               WRID_EAGER_SLOT_BASE +
               PG_EAGER_RECV_SLOTS) {

        int slot_index =
            (int)(
                wc[i].wr_id -
                WRID_EAGER_SLOT_BASE);

        if (!have_recv) {
          have_recv = 1;
          recv_slot = slot_index;
        }
        else {
          if (save_eager_completed_slot(
              pg,
              slot_index) != 0) {
            return -1;
          }
        }
      }
      else {
        fprintf(stderr,
                "Rank %d: unexpected completion wr_id=%llu "
                "while waiting for Eager exchange\n",
                pg->cfg.rank,
                (unsigned long long)wc[i].wr_id);

        return -1;
      }
    }
  }

  *completed_recv_slot = recv_slot;
  return 0;
}

static int post_rendezvous_write_and_flag(
    pg_handle_t *pg,
    const void *send_buf,
    struct ibv_mr *send_mr,
    uint64_t remote_data_addr,
    uint32_t remote_data_rkey,
    size_t bytes,
    int step,
    uint32_t epoch)
{
  if (!pg || !send_buf || !send_mr ||
      bytes == 0 || bytes > UINT32_MAX ||
      step < 0 || step >= PG_MAX_HOSTS) {
    return -1;
  }

  pg->rendezvous_control.signal_value = epoch;

  struct ibv_sge data_sge = {
      .addr = (uintptr_t)send_buf,
      .length = (uint32_t)bytes,
      .lkey = send_mr->lkey
  };

  struct ibv_sge flag_sge = {
      .addr = (uintptr_t)&pg->rendezvous_control.signal_value,
      .length = sizeof(uint32_t),
      .lkey = pg->rendezvous_control_mr->lkey
  };

  struct ibv_send_wr data_wr;
  struct ibv_send_wr flag_wr;
  memset(&data_wr, 0, sizeof(data_wr));
  memset(&flag_wr, 0, sizeof(flag_wr));

  /* First write the data. This WR is intentionally unsignaled. */
  data_wr.wr_id = WRID_WRITE_SEND;
  data_wr.sg_list = &data_sge;
  data_wr.num_sge = 1;
  data_wr.opcode = IBV_WR_RDMA_WRITE;
  data_wr.send_flags = 0;
  data_wr.wr.rdma.remote_addr = remote_data_addr;
  data_wr.wr.rdma.rkey = remote_data_rkey;
  data_wr.next = &flag_wr;

  /*
   * Then write the epoch flag. RC QP ordering guarantees that the
   * remote data write is visible before this flag becomes visible.
   */
  flag_wr.wr_id = WRID_WRITE_SEND;
  flag_wr.sg_list = &flag_sge;
  flag_wr.num_sge = 1;
  flag_wr.opcode = IBV_WR_RDMA_WRITE;
  flag_wr.send_flags = IBV_SEND_SIGNALED;
  flag_wr.wr.rdma.remote_addr =
      pg->next_remote_control_mr.address +
      offsetof(rendezvous_control_t, flags) +
      (uint64_t)step * sizeof(uint32_t);
  flag_wr.wr.rdma.rkey = pg->next_remote_control_mr.rkey;
  flag_wr.next = NULL;

  struct ibv_send_wr *bad_send_wr = NULL;

  if (ibv_post_send(pg->qp_next, &data_wr, &bad_send_wr) != 0) {
    fprintf(stderr,
            "Rank %d: failed posting Rendezvous data/flag writes: %s\n",
            pg->cfg.rank,
            strerror(errno));
    return -1;
  }

  return 0;
}

static int wait_for_rendezvous_flag(
    pg_handle_t *pg,
    int step,
    uint32_t epoch)
{
  if (!pg || step < 0 || step >= PG_MAX_HOSTS) {
    return -1;
  }

  volatile uint32_t *flag =
      &pg->rendezvous_control.flags[step];

  while (*flag != epoch) {
    /* Prevent the compiler from caching the load. */
    __asm__ __volatile__("" ::: "memory");
  }

  return 0;
}

/* ------------------------------------------------------------------------- */
/* Pipelined Collectives Implementation                                      */
/* ------------------------------------------------------------------------- */

/* Pipelined Reduce-Scatter (Overlaps networking and reduction math) */
static int pg_reduce_scatter_pipelined(
    pg_handle_t *pg,
    uint8_t *work_buf,
    struct ibv_mr *work_mr,
    size_t chunk_elements,
    size_t chunk_bytes,
    DATATYPE datatype,
    OPERATION operation,
    int *owned_chunk)
{
  int P = pg->cfg.size;
  int rank = pg->cfg.rank;

  size_t elem_size = datatype_size(datatype);

  size_t micro_chunk_elems = PG_MICRO_CHUNK_SIZE;
  if (micro_chunk_elems > chunk_elements) {
    micro_chunk_elems = chunk_elements;
  }

  size_t micro_chunk_bytes = micro_chunk_elems * elem_size;
  if (micro_chunk_bytes == 0) {
    return -1;
  }

  /*
   * The receive queue was filled during connect_process_group().
   * Each slot is reposted immediately after its completion is consumed.
   */
  int expected_slot = pg->next_expected_eager_slot;

  for (int step = 0; step < P - 1; ++step) {
    int send_idx = (rank - step + P) % P;
    int recv_idx = (rank - step - 1 + P) % P;

    uint8_t *src_ptr =
        work_buf + (size_t)send_idx * chunk_bytes;

    uint8_t *dst_ptr =
        work_buf + (size_t)recv_idx * chunk_bytes;

    /*
     * Prime the pipeline:
     * post the first micro-chunk before doing any reduction.
     */
    size_t current_offset = 0;

    size_t current_bytes =
        chunk_bytes > micro_chunk_bytes
        ? micro_chunk_bytes
        : chunk_bytes;

    if (post_eager_send(
        pg,
        src_ptr + current_offset,
        work_mr,
        current_bytes) != 0) {
      return -1;
    }

    while (current_offset < chunk_bytes) {
      int completed_slot = -1;

      /*
       * Wait until the current outgoing micro-chunk has completed
       * and the corresponding incoming micro-chunk is available.
       */
      if (wait_for_eager_pair(
          pg,
          &completed_slot) != 0) {
        return -1;
      }

      if (completed_slot != expected_slot) {
        fprintf(stderr,
                "Rank %d: unexpected Eager receive order: "
                "got slot %d, expected slot %d\n",
                pg->cfg.rank,
                completed_slot,
                expected_slot);
        return -1;
      }

      uint8_t *received_data =
          (uint8_t *)pg->eager_recv_pool +
          (size_t)completed_slot * pg->eager_slot_size;

      size_t current_elems =
          current_bytes / elem_size;

      /*
       * Start communication for the NEXT micro-chunk before
       * reducing the CURRENT one.
       *
       * The RNIC can now transfer the next micro-chunk while
       * the CPU performs reduce_local() below.
       */
      size_t next_offset =
          current_offset + current_bytes;

      if (next_offset < chunk_bytes) {
        size_t remaining =
            chunk_bytes - next_offset;

        size_t next_bytes =
            remaining > micro_chunk_bytes
            ? micro_chunk_bytes
            : remaining;

        if (post_eager_send(
            pg,
            src_ptr + next_offset,
            work_mr,
            next_bytes) != 0) {
          return -1;
        }
      }

      /*
       * CPU computation overlaps with the communication
       * posted above.
       */
      if (reduce_local(
          dst_ptr + current_offset,
          received_data,
          current_elems,
          datatype,
          operation) != 0) {
        return -1;
      }

      /*
       * The completed receive buffer can now be reused.
       */
      if (post_eager_receive_slot(
          pg,
          completed_slot,
          pg->eager_slot_size) != 0) {
        return -1;
      }

      expected_slot =
          (expected_slot + 1) %
          PG_EAGER_RECV_SLOTS;

      pg->next_expected_eager_slot =
          expected_slot;

      current_offset = next_offset;

      if (current_offset < chunk_bytes) {
        size_t remaining =
            chunk_bytes - current_offset;

        current_bytes =
            remaining > micro_chunk_bytes
            ? micro_chunk_bytes
            : remaining;
      }
    }
  }

  *owned_chunk = (rank + 1) % P;
  return 0;
}

/* Ring All-Gather */
static int pg_all_gather_pipelined(
    pg_handle_t *pg,
    uint8_t *work_buf,
    struct ibv_mr *work_mr,
    const remote_mr_info_t *next_remote,
    size_t chunk_bytes,
    int owned_chunk,
    uint32_t rendezvous_epoch)
{
  int P = pg->cfg.size;
  int expected_slot = pg->next_expected_eager_slot;

  for (int step = 0; step < P - 1; ++step) {
    int send_idx =
        (owned_chunk - step + P) % P;

    int recv_idx =
        (owned_chunk - step - 1 + P) % P;

    uint8_t *send_chunk =
        work_buf + (size_t)send_idx * chunk_bytes;

    uint8_t *recv_chunk =
        work_buf + (size_t)recv_idx * chunk_bytes;

    if (pg->cfg.protocol == PROTOCOL_RENDEZVOUS) {
      uint64_t remote_dest =
          next_remote->address +
          (uint64_t)send_idx * chunk_bytes;

      if (post_rendezvous_write_and_flag(
          pg,
          send_chunk,
          work_mr,
          remote_dest,
          next_remote->rkey,
          chunk_bytes,
          step,
          rendezvous_epoch) != 0) {
        return -1;
      }

      /* Wait for the previous rank's data-ready flag. */
      if (wait_for_rendezvous_flag(
          pg,
          step,
          rendezvous_epoch) != 0) {
        return -1;
      }

      /* Wait for the local signaled flag-write completion. */
      if (wait_for_completions(pg, 1) != 0) {
        return -1;
      }
    }
    else {
      if (post_eager_send(
          pg,
          send_chunk,
          work_mr,
          chunk_bytes) != 0) {
        return -1;
      }

      int completed_slot = -1;

      if (wait_for_eager_pair(pg, &completed_slot) != 0) {
        return -1;
      }

      if (completed_slot != expected_slot) {
        fprintf(stderr,
                "Rank %d: unexpected Eager All-Gather "
                "receive slot: got %d, expected %d\n",
                pg->cfg.rank,
                completed_slot,
                expected_slot);
        return -1;
      }

      uint8_t *received_data =
          (uint8_t *)pg->eager_recv_pool +
          (size_t)completed_slot * pg->eager_slot_size;

      memcpy(recv_chunk, received_data, chunk_bytes);

      if (post_eager_receive_slot(
          pg,
          completed_slot,
          pg->eager_slot_size) != 0) {
        return -1;
      }

      expected_slot =
          (expected_slot + 1) % PG_EAGER_RECV_SLOTS;
      pg->next_expected_eager_slot = expected_slot;
    }
  }

  return 0;
}

/* ------------------------------------------------------------------------- */
/* Public API Implementations                                                */
/* ------------------------------------------------------------------------- */

static void pg_release_work_buffer(pg_handle_t *pg)
{
  if (!pg) return;

  if (pg->work_mr) {
    ibv_dereg_mr(pg->work_mr);
    pg->work_mr = NULL;
  }

  pg->work_buffer = NULL;
  pg->work_buffer_bytes = 0;
  memset(&pg->next_remote_mr, 0, sizeof(pg->next_remote_mr));
}

static int pg_prepare_work_buffer(pg_handle_t *pg,
                                  void *recvbuf,
                                  size_t total_bytes)
{
  if (!pg || !recvbuf || total_bytes == 0 || total_bytes > UINT32_MAX) {
    return -1;
  }

  if (pg->work_mr &&
      pg->work_buffer == recvbuf &&
      pg->work_buffer_bytes == total_bytes) {
    return 0;
  }

  pg_release_work_buffer(pg);

  pg->work_mr = ibv_reg_mr(
      pg->pd,
      recvbuf,
      total_bytes,
      IBV_ACCESS_LOCAL_WRITE |
      IBV_ACCESS_REMOTE_WRITE);

  if (!pg->work_mr) {
    fprintf(stderr,
            "Rank %d: ibv_reg_mr failed for %zu bytes: %s\n",
            pg->cfg.rank,
            total_bytes,
            strerror(errno));
    return -1;
  }

  wire_mr_info_t local_wire = {
      .address = htobe64((uint64_t)(uintptr_t)recvbuf),
      .rkey = htonl(pg->work_mr->rkey),
      .bytes = htonl((uint32_t)total_bytes)
  };

  wire_mr_info_t remote_wire;
  memset(&remote_wire, 0, sizeof(remote_wire));

  if (send_all(pg->socket_from_prev,
               &local_wire,
               sizeof(local_wire)) != 0 ||
      recv_all(pg->socket_to_next,
               &remote_wire,
               sizeof(remote_wire)) != 0) {
    fprintf(stderr,
            "Rank %d: failed exchanging work-buffer MR information\n",
            pg->cfg.rank);
    pg_release_work_buffer(pg);
    return -1;
  }

  pg->next_remote_mr.address = be64toh(remote_wire.address);
  pg->next_remote_mr.rkey = ntohl(remote_wire.rkey);
  pg->next_remote_mr.bytes = ntohl(remote_wire.bytes);

  if (pg->next_remote_mr.bytes < total_bytes) {
    fprintf(stderr,
            "Rank %d: next rank's MR is too small: "
            "remote=%u bytes, required=%zu bytes\n",
            pg->cfg.rank,
            pg->next_remote_mr.bytes,
            total_bytes);
    pg_release_work_buffer(pg);
    return -1;
  }

  pg->work_buffer = recvbuf;
  pg->work_buffer_bytes = total_bytes;
  return 0;
}

int pg_all_reduce(void *sendbuf, void *recvbuf, int count,
                  DATATYPE datatype, OPERATION operation, void *pg_handle)
{
  pg_handle_t *pg = (pg_handle_t *)pg_handle;

  if (!pg || !sendbuf || !recvbuf || count <= 0) {
    return -1;
  }

  size_t elem_size = datatype_size(datatype);
  if (elem_size == 0 || (size_t)count > SIZE_MAX / elem_size) {
    return -1;
  }

  size_t total_bytes = (size_t)count * elem_size;

  if ((size_t)count % (size_t)pg->cfg.size != 0) {
    fprintf(stderr,
            "Rank %d: element count must be divisible by process count\n",
            pg->cfg.rank);
    return -1;
  }

  size_t chunk_bytes = total_bytes / (size_t)pg->cfg.size;
  size_t chunk_elems = (size_t)count / (size_t)pg->cfg.size;

  if (pg->cfg.protocol == PROTOCOL_EAGER &&
      chunk_bytes > pg->eager_slot_size) {
    fprintf(stderr,
            "Rank %d: eager chunk size %zu exceeds "
            "Eager receive slot size %zu\n",
            pg->cfg.rank,
            chunk_bytes,
            pg->eager_slot_size);
    return -1;
  }

  /* Normally prepared by main before timing; retained as API safety. */
  if (pg_prepare_work_buffer(pg, recvbuf, total_bytes) != 0) {
    return -1;
  }

  memcpy(recvbuf, sendbuf, total_bytes);

  int owned_chunk = -1;

  /*
   * Each call uses a new generation value for one-sided Rendezvous
   * completion flags. No flag reset or phase barrier is required.
   */
  ++pg->rendezvous_epoch;
  if (pg->rendezvous_epoch == 0) {
    pg->rendezvous_epoch = 1;
  }

  if (pg_reduce_scatter_pipelined(
      pg,
      recvbuf,
      pg->work_mr,
      chunk_elems,
      chunk_bytes,
      datatype,
      operation,
      &owned_chunk) != 0) {
    return -1;
  }

  if (pg_all_gather_pipelined(
      pg,
      recvbuf,
      pg->work_mr,
      &pg->next_remote_mr,
      chunk_bytes,
      owned_chunk,
      pg->rendezvous_epoch) != 0) {
    return -1;
  }

  return 0;
}

static int parse_rank(const char *text) {
  long val = strtol(text, NULL, 10);
  if (text[0] == '0' && text[1] != '\0') --val;
  return (val >= 0) ? (int)val : -1;
}

static int create_listener(int port)
{
  int socket_fd = socket(AF_INET, SOCK_STREAM, 0);

  if (socket_fd < 0) {
    perror("socket");
    return -1;
  }

  int one = 1;

  if (setsockopt(socket_fd,
                 SOL_SOCKET,
                 SO_REUSEADDR,
                 &one,
                 sizeof(one)) != 0) {
    perror("setsockopt SO_REUSEADDR");
    close(socket_fd);
    return -1;
  }

  struct sockaddr_in addr;
  memset(&addr, 0, sizeof(addr));

  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_ANY);
  addr.sin_port = htons((uint16_t)port);

  if (bind(socket_fd,
           (struct sockaddr *)&addr,
           sizeof(addr)) != 0) {
    perror("bind");
    close(socket_fd);
    return -1;
  }

  if (listen(socket_fd, 8) != 0) {
    perror("listen");
    close(socket_fd);
    return -1;
  }

  return socket_fd;
}

static int connect_with_retry(const char *hostname, int port)
{
  char port_string[16];

  snprintf(
      port_string,
      sizeof(port_string),
      "%d",
      port
  );

  struct addrinfo hints;
  memset(&hints, 0, sizeof(hints));

  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;

  struct addrinfo *results = NULL;

  int gai_rc = getaddrinfo(
      hostname,
      port_string,
      &hints,
      &results
  );

  if (gai_rc != 0) {
    fprintf(stderr,
            "getaddrinfo(%s): %s\n",
            hostname,
            gai_strerror(gai_rc));
    return -1;
  }

  for (int attempt = 0; attempt < 100; ++attempt) {
    for (struct addrinfo *entry = results;
         entry != NULL;
         entry = entry->ai_next) {

      int socket_fd = socket(
          entry->ai_family,
          entry->ai_socktype,
          entry->ai_protocol
      );

      if (socket_fd < 0)
        continue;

      if (connect(
          socket_fd,
          entry->ai_addr,
          entry->ai_addrlen) == 0) {

        freeaddrinfo(results);
        return socket_fd;
      }

      close(socket_fd);
    }

    struct timespec delay = {
        .tv_sec = 0,
        .tv_nsec = 100000000
    };

    nanosleep(&delay, NULL);
  }

  freeaddrinfo(results);
  return -1;
}

static void qp_info_to_wire(const qp_info_t *host,
                            wire_qp_info_t *wire)
{
  memset(wire, 0, sizeof(*wire));

  wire->lid = htons(host->lid);
  wire->qpn = htonl(host->qpn);
  wire->psn = htonl(host->psn);
  wire->mtu = (uint8_t)host->mtu;

  memcpy(wire->gid,
         host->gid.raw,
         sizeof(wire->gid));
}

static void qp_info_from_wire(const wire_qp_info_t *wire,
                              qp_info_t *host)
{
  memset(host, 0, sizeof(*host));

  host->lid = ntohs(wire->lid);
  host->qpn = ntohl(wire->qpn);
  host->psn = ntohl(wire->psn);
  host->mtu = (enum ibv_mtu)wire->mtu;

  memcpy(host->gid.raw,
         wire->gid,
         sizeof(wire->gid));
}

static struct ibv_qp *create_rc_qp(pg_handle_t *pg)
{
  struct ibv_qp_init_attr init_attr;
  memset(&init_attr, 0, sizeof(init_attr));

  init_attr.qp_type = IBV_QPT_RC;

  init_attr.send_cq = pg->cq;
  init_attr.recv_cq = pg->cq;

  init_attr.cap.max_send_wr = PG_MAX_WR;
  init_attr.cap.max_recv_wr = PG_MAX_WR;

  init_attr.cap.max_send_sge = 1;
  init_attr.cap.max_recv_sge = 1;

  init_attr.sq_sig_all = 0;

  return ibv_create_qp(pg->pd, &init_attr);
}

static int modify_qp_to_init(pg_handle_t *pg,
                             struct ibv_qp *qp)
{
  struct ibv_qp_attr attr;
  memset(&attr, 0, sizeof(attr));

  attr.qp_state = IBV_QPS_INIT;
  attr.port_num = pg->cfg.ib_port;
  attr.pkey_index = 0;

  attr.qp_access_flags =
      IBV_ACCESS_LOCAL_WRITE |
      IBV_ACCESS_REMOTE_WRITE;

  int mask =
      IBV_QP_STATE |
      IBV_QP_PKEY_INDEX |
      IBV_QP_PORT |
      IBV_QP_ACCESS_FLAGS;

  if (ibv_modify_qp(qp, &attr, mask) != 0) {
    fprintf(stderr,
            "Rank %d: failed moving QP %u to INIT: %s\n",
            pg->cfg.rank,
            qp->qp_num,
            strerror(errno));

    return -1;
  }

  return 0;
}

static int modify_qp_to_rtr(pg_handle_t *pg,
                            struct ibv_qp *qp,
                            const qp_info_t *remote)
{
  struct ibv_qp_attr attr;
  memset(&attr, 0, sizeof(attr));

  attr.qp_state = IBV_QPS_RTR;

/*
 * Use the largest MTU supported by both endpoints.
 */
  enum ibv_mtu local_mtu = pg->port_attr.active_mtu;
  enum ibv_mtu remote_mtu = remote->mtu;

  if (remote_mtu < IBV_MTU_256 ||
      remote_mtu > IBV_MTU_4096) {
    fprintf(stderr,
            "Rank %d: invalid remote MTU value %d\n",
            pg->cfg.rank,
            (int)remote_mtu);
    return -1;
  }

  attr.path_mtu =
      (local_mtu < remote_mtu)
      ? local_mtu
      : remote_mtu;

  attr.dest_qp_num = remote->qpn;
  attr.rq_psn = remote->psn;

  attr.max_dest_rd_atomic = 1;
  attr.min_rnr_timer = 12; //how long the sender waits after an RNR
  // (Receiver Not Ready) before retrying

  attr.ah_attr.port_num = pg->cfg.ib_port;
  attr.ah_attr.sl = 0;
  attr.ah_attr.src_path_bits = 0;

  if (pg->port_attr.link_layer ==
      IBV_LINK_LAYER_ETHERNET) {

    /*
     * RoCE uses GID routing.
     */
    attr.ah_attr.is_global = 1;
    attr.ah_attr.dlid = 0;

    attr.ah_attr.grh.dgid =
        remote->gid;

    attr.ah_attr.grh.sgid_index =
        pg->cfg.gid_index;

    attr.ah_attr.grh.hop_limit = 64;
    attr.ah_attr.grh.traffic_class = 0;
    attr.ah_attr.grh.flow_label = 0;
  }
  else {
    /*
     * Native InfiniBand uses LID routing.
     */
    if (remote->lid == 0) {
      fprintf(stderr,
              "Rank %d: remote LID is zero for QP %u\n",
              pg->cfg.rank,
              qp->qp_num);
      return -1;
    }

    attr.ah_attr.is_global = 0;
    attr.ah_attr.dlid = remote->lid;
  }

  int mask =
      IBV_QP_STATE |
      IBV_QP_AV |
      IBV_QP_PATH_MTU |
      IBV_QP_DEST_QPN |
      IBV_QP_RQ_PSN |
      IBV_QP_MAX_DEST_RD_ATOMIC |
      IBV_QP_MIN_RNR_TIMER;

  if (ibv_modify_qp(qp,
                    &attr,
                    mask) != 0) {

    fprintf(stderr,
            "Rank %d: failed moving QP %u to RTR: %s\n",
            pg->cfg.rank,
            qp->qp_num,
            strerror(errno));

    return -1;
  }

  fprintf(stderr,
          "Rank %d: QP %u moved to RTR using %s routing, "
          "remote LID=%u, remote QPN=%u, remote PSN=%u\n",
          pg->cfg.rank,
          qp->qp_num,
          pg->port_attr.link_layer ==
          IBV_LINK_LAYER_ETHERNET
          ? "GID"
          : "LID",
          remote->lid,
          remote->qpn,
          remote->psn);

  return 0;
}

static int modify_qp_to_rts(pg_handle_t *pg,
                            struct ibv_qp *qp,
                            uint32_t local_psn)
{
  struct ibv_qp_attr attr;
  memset(&attr, 0, sizeof(attr));

  attr.qp_state = IBV_QPS_RTS;

  attr.timeout = 14;
  attr.retry_cnt = 7;
  attr.rnr_retry = 7;

  attr.sq_psn = local_psn;
  attr.max_rd_atomic = 1;

  int mask =
      IBV_QP_STATE |
      IBV_QP_TIMEOUT |
      IBV_QP_RETRY_CNT |
      IBV_QP_RNR_RETRY |
      IBV_QP_SQ_PSN |
      IBV_QP_MAX_QP_RD_ATOMIC;

  if (ibv_modify_qp(qp, &attr, mask) != 0) {
    fprintf(stderr,
            "Rank %d: failed moving QP %u to RTS: %s\n",
            pg->cfg.rank,
            qp->qp_num,
            strerror(errno));

    return -1;
  }

  return 0;
}

static uint32_t create_local_psn(int rank,
                                 int qp_number)
{
  /*
   * Generate different 24-bit PSNs for the two QPs.
   * RC PSNs use only 24 bits.
   */
  uint32_t value =
      0x12345u +
      (uint32_t)rank * 0x101u +
      (uint32_t)qp_number * 0x31u;

  return value & 0x00ffffffu;
}

static void print_gid(const union ibv_gid *gid)
{
  char text[INET6_ADDRSTRLEN];

  if (inet_ntop(AF_INET6,
                gid->raw,
                text,
                sizeof(text))) {
    fprintf(stderr, "%s", text);
  }
  else {
    fprintf(stderr, "<invalid GID>");
  }
}

static int exchange_qp_information(pg_handle_t *pg,
                                   uint32_t next_local_psn,
                                   uint32_t prev_local_psn)
{
  qp_info_t local_next;
  qp_info_t local_prev;

  memset(&local_next, 0, sizeof(local_next));
  memset(&local_prev, 0, sizeof(local_prev));

  local_next.lid = pg->port_attr.lid;
  local_next.qpn = pg->qp_next->qp_num;
  local_next.psn = next_local_psn;
  local_next.gid = pg->local_gid;
  local_next.mtu = pg->port_attr.active_mtu;

  local_prev.lid = pg->port_attr.lid;
  local_prev.qpn = pg->qp_prev->qp_num;
  local_prev.psn = prev_local_psn;
  local_prev.gid = pg->local_gid;
  local_prev.mtu = pg->port_attr.active_mtu;

  wire_qp_info_t local_next_wire;
  wire_qp_info_t local_prev_wire;

  wire_qp_info_t remote_next_wire;
  wire_qp_info_t remote_prev_wire;

  qp_info_to_wire(&local_next, &local_next_wire);
  qp_info_to_wire(&local_prev, &local_prev_wire);

  /*
   * Edge rank -> next:
   *
   * local qp_next connects to next rank's qp_prev.
   *
   * Edge prev -> rank:
   *
   * local qp_prev connects to previous rank's qp_next.
   */

  if (send_all(pg->socket_to_next,
               &local_next_wire,
               sizeof(local_next_wire)) != 0) {
    fprintf(stderr,
            "Rank %d: failed sending qp_next information\n",
            pg->cfg.rank);
    return -1;
  }

  if (send_all(pg->socket_from_prev,
               &local_prev_wire,
               sizeof(local_prev_wire)) != 0) {
    fprintf(stderr,
            "Rank %d: failed sending qp_prev information\n",
            pg->cfg.rank);
    return -1;
  }

  /*
   * Receive the next process's qp_prev information.
   */
  if (recv_all(pg->socket_to_next,
               &remote_next_wire,
               sizeof(remote_next_wire)) != 0) {
    fprintf(stderr,
            "Rank %d: failed receiving next qp information\n",
            pg->cfg.rank);
    return -1;
  }

  /*
   * Receive the previous process's qp_next information.
   */
  if (recv_all(pg->socket_from_prev,
               &remote_prev_wire,
               sizeof(remote_prev_wire)) != 0) {
    fprintf(stderr,
            "Rank %d: failed receiving previous qp information\n",
            pg->cfg.rank);
    return -1;
  }

  qp_info_from_wire(&remote_next_wire,
                    &pg->next_qp_info);

  qp_info_from_wire(&remote_prev_wire,
                    &pg->prev_qp_info);

  fprintf(stderr,
          "Rank %d: local qp_next=%u, remote next qp=%u\n",
          pg->cfg.rank,
          pg->qp_next->qp_num,
          pg->next_qp_info.qpn);

  fprintf(stderr,
          "Rank %d: local qp_prev=%u, remote prev qp=%u\n",
          pg->cfg.rank,
          pg->qp_prev->qp_num,
          pg->prev_qp_info.qpn);

  fprintf(stderr,
          "Rank %d: next remote LID=%u QPN=%u PSN=%u GID=",
          pg->cfg.rank,
          pg->next_qp_info.lid,
          pg->next_qp_info.qpn,
          pg->next_qp_info.psn);

  print_gid(&pg->next_qp_info.gid);
  fprintf(stderr, "\n");

  fprintf(stderr,
          "Rank %d: previous remote LID=%u QPN=%u PSN=%u GID=",
          pg->cfg.rank,
          pg->prev_qp_info.lid,
          pg->prev_qp_info.qpn,
          pg->prev_qp_info.psn);

  print_gid(&pg->prev_qp_info.gid);
  fprintf(stderr, "\n");

  return 0;
}

static bool gid_is_all_zero(const union ibv_gid *gid)
{
  static const uint8_t zero[16] = {0};

  return memcmp(gid->raw, zero, sizeof(zero)) == 0;
}

static int choose_gid_index(pg_handle_t *pg)
{
  int table_length = pg->port_attr.gid_tbl_len;

  if (table_length <= 0) {
    fprintf(stderr,
            "Rank %d: RDMA port has no GID entries\n",
            pg->cfg.rank);
    return -1;
  }

  fprintf(stderr,
          "Rank %d: available GIDs on port %d:\n",
          pg->cfg.rank,
          pg->cfg.ib_port);

  int selected_index = -1;
  union ibv_gid selected_gid;
  memset(&selected_gid, 0, sizeof(selected_gid));

  for (int index = 0;
       index < table_length;
       ++index) {

    union ibv_gid gid;
    memset(&gid, 0, sizeof(gid));

    if (ibv_query_gid(pg->context,
                      pg->cfg.ib_port,
                      index,
                      &gid) != 0) {
      continue;
    }

    fprintf(stderr, "  GID index %d: ", index);
    print_gid(&gid);

    if (gid_is_all_zero(&gid)) {
      fprintf(stderr, " [zero]\n");
      continue;
    }

    fprintf(stderr, "\n");

    /*
     * Prefer an IPv4-mapped GID:
     *
     * ::ffff:a.b.c.d
     *
     * These are commonly the correct RoCEv2 entries.
     */
    bool ipv4_mapped =
        gid.raw[0] == 0x00 &&
        gid.raw[1] == 0x00 &&
        gid.raw[2] == 0x00 &&
        gid.raw[3] == 0x00 &&
        gid.raw[4] == 0x00 &&
        gid.raw[5] == 0x00 &&
        gid.raw[6] == 0x00 &&
        gid.raw[7] == 0x00 &&
        gid.raw[8] == 0x00 &&
        gid.raw[9] == 0x00 &&
        gid.raw[10] == 0xff &&
        gid.raw[11] == 0xff;

    if (ipv4_mapped) {
      selected_index = index;
      selected_gid = gid;
      break;
    }

    /*
     * Keep the first non-zero GID as a fallback.
     */
    if (selected_index < 0) {
      selected_index = index;
      selected_gid = gid;
    }
  }

  if (selected_index < 0) {
    fprintf(stderr,
            "Rank %d: could not find a non-zero GID\n",
            pg->cfg.rank);
    return -1;
  }

  pg->cfg.gid_index = selected_index;
  pg->local_gid = selected_gid;

  fprintf(stderr,
          "Rank %d: selected GID index %d: ",
          pg->cfg.rank,
          selected_index);

  print_gid(&pg->local_gid);
  fprintf(stderr, "\n");

  return 0;
}

int connect_process_group(char *servername,
                          void **pg_handle)
{
  (void)servername;

  if (!pg_handle) {
    fprintf(stderr,
            "Rank %d: pg_handle is NULL\n",
            g_config.rank);
    return -1;
  }

  *pg_handle = NULL;

  pg_handle_t *pg =
      calloc(1, sizeof(*pg));

  if (!pg) {
    perror("calloc");
    return -1;
  }

  pg->cfg = g_config;

  pg->socket_to_next = -1;
  pg->socket_from_prev = -1;
  pg->work_buffer = NULL;
  pg->work_buffer_bytes = 0;
  pg->work_mr = NULL;
  memset(&pg->next_remote_mr, 0, sizeof(pg->next_remote_mr));
  pg->next_expected_eager_slot = 0;
  pg->rendezvous_epoch = 0;
  memset(&pg->rendezvous_control, 0, sizeof(pg->rendezvous_control));
  pg->rendezvous_control_mr = NULL;
  memset(&pg->next_remote_control_mr,
         0,
         sizeof(pg->next_remote_control_mr));

  int listener = -1;
  int num_devices = 0;

  /* -------------------------------------------------------------- */
  /* Open RDMA device                                               */
  /* -------------------------------------------------------------- */

  // It returns a list of RDMA devices that the verbs API can work with.
  pg->device_list =
      ibv_get_device_list(&num_devices);

  if (!pg->device_list) {
    fprintf(stderr,
            "Rank %d: ibv_get_device_list failed: %s\n",
            pg->cfg.rank,
            strerror(errno));
    goto error;
  }

  if (num_devices <= 0) {
    fprintf(stderr,
            "Rank %d: no RDMA devices found\n",
            pg->cfg.rank);
    goto error;
  }

  fprintf(stderr,
          "Rank %d: using RDMA device %s\n",
          pg->cfg.rank,
          ibv_get_device_name(pg->device_list[0]));

  // Open the first available RDMA device and obtain a context for Verbs
  // operations.
  pg->context =
      ibv_open_device(pg->device_list[0]);

  if (!pg->context) {
    fprintf(stderr,
            "Rank %d: ibv_open_device failed: %s\n",
            pg->cfg.rank,
            strerror(errno));
    goto error;
  }

  /* -------------------------------------------------------------- */
  /* Query port and GID                                             */
  /* -------------------------------------------------------------- */

  if (ibv_query_port(pg->context,
                     pg->cfg.ib_port,
                     &pg->port_attr) != 0) {
    fprintf(stderr,
            "Rank %d: ibv_query_port failed: %s\n",
            pg->cfg.rank,
            strerror(errno));
    goto error;
  }

  fprintf(stderr,
          "Rank %d: RDMA port %d link layer: %s, "
          "local LID=%u, active MTU=%d\n",
          pg->cfg.rank,
          pg->cfg.ib_port,
          pg->port_attr.link_layer ==
          IBV_LINK_LAYER_ETHERNET
          ? "Ethernet/RoCE"
          : "InfiniBand",
          pg->port_attr.lid,
          pg->port_attr.active_mtu);

  if (pg->port_attr.state != IBV_PORT_ACTIVE) {
    fprintf(stderr,
            "Rank %d: RDMA port %d is not active; state=%d\n",
            pg->cfg.rank,
            pg->cfg.ib_port,
            pg->port_attr.state);
    goto error;
  }

  memset(&pg->local_gid,
         0,
         sizeof(pg->local_gid));

  if (pg->port_attr.link_layer ==
      IBV_LINK_LAYER_ETHERNET) {

    if (choose_gid_index(pg) != 0) {
      goto error;
    }
  }
  else {
    /*
     * Native InfiniBand primarily uses LIDs.
     * The GID can still be exchanged, but it must not force
     * global routing in modify_qp_to_rtr().
     */
    if (ibv_query_gid(pg->context,
                      pg->cfg.ib_port,
                      pg->cfg.gid_index,
                      &pg->local_gid) != 0) {

      memset(&pg->local_gid,
             0,
             sizeof(pg->local_gid));
    }
  }

  /* -------------------------------------------------------------- */
  /* Create PD and CQ                                               */
  /* -------------------------------------------------------------- */

  /* Allocate a Protection Domain for this RDMA process. */
  pg->pd = ibv_alloc_pd(pg->context);

  if (!pg->pd) {
    fprintf(stderr,
            "Rank %d: ibv_alloc_pd failed: %s\n",
            pg->cfg.rank,
            strerror(errno));
    goto error;
  }

  /* Create a Completion Queue for completed RDMA operations. */
  pg->cq = ibv_create_cq(
      pg->context,
      PG_CQ_SIZE,
      NULL,
      NULL,
      0
  );

  if (!pg->cq) {
    fprintf(stderr,
            "Rank %d: ibv_create_cq failed: %s\n",
            pg->cfg.rank,
            strerror(errno));
    goto error;
  }

  /* -------------------------------------------------------------- */
  /* Create queue pairs                                             */
  /* -------------------------------------------------------------- */

  /* Create the QP used to communicate with the next rank in the ring. */
  pg->qp_next = create_rc_qp(pg);

  if (!pg->qp_next) {
    fprintf(stderr,
            "Rank %d: failed creating qp_next: %s\n",
            pg->cfg.rank,
            strerror(errno));
    goto error;
  }

  /* Create the QP used to communicate with the previous rank in the ring. */
  pg->qp_prev = create_rc_qp(pg);

  if (!pg->qp_prev) {
    fprintf(stderr,
            "Rank %d: failed creating qp_prev: %s\n",
            pg->cfg.rank,
            strerror(errno));
    goto error;
  }

  /* Move both QPs to INIT state before connecting them to remote QPs. */
  if (modify_qp_to_init(pg, pg->qp_next) != 0 ||
      modify_qp_to_init(pg, pg->qp_prev) != 0) {
    goto error;
  }

  /* -------------------------------------------------------------- */
  /* Configure Eager receive slot size                              */
  /* -------------------------------------------------------------- */

  /*
   * Each pre-posted Eager receive uses a slot of this size.
   * The value is large enough to hold one full All-Gather chunk
   * for the supported benchmark message sizes.
   */
  pg->eager_slot_size =
      PG_MICRO_CHUNK_SIZE *
      sizeof(double) *
      4;

  /* -------------------------------------------------------------- */
  /* Allocate registered Eager receive pool                          */
  /* -------------------------------------------------------------- */

  size_t eager_pool_bytes =
      (size_t)PG_EAGER_RECV_SLOTS *
      pg->eager_slot_size;

  int eager_alloc_rc =
      posix_memalign(
          &pg->eager_recv_pool,
          64,
          eager_pool_bytes);

  if (eager_alloc_rc != 0) {
    fprintf(stderr,
            "Rank %d: posix_memalign for Eager receive pool "
            "failed: %s\n",
            pg->cfg.rank,
            strerror(eager_alloc_rc));
    goto error;
  }

  memset(
      pg->eager_recv_pool,
      0,
      eager_pool_bytes);

  pg->eager_recv_mr =
      ibv_reg_mr(
          pg->pd,
          pg->eager_recv_pool,
          eager_pool_bytes,
          IBV_ACCESS_LOCAL_WRITE);

  if (!pg->eager_recv_mr) {
    fprintf(stderr,
            "Rank %d: ibv_reg_mr for Eager receive pool "
            "failed: %s\n",
            pg->cfg.rank,
            strerror(errno));
    goto error;
  }

  pg->rendezvous_control_mr = ibv_reg_mr(
      pg->pd,
      &pg->rendezvous_control,
      sizeof(pg->rendezvous_control),
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);

  if (!pg->rendezvous_control_mr) {
    fprintf(stderr,
            "Rank %d: ibv_reg_mr for Rendezvous control failed: %s\n",
            pg->cfg.rank,
            strerror(errno));
    goto error;
  }

  /* -------------------------------------------------------------- */
  /* Create TCP ring                                                */
  /* -------------------------------------------------------------- */

  int next_rank =
      (pg->cfg.rank + 1) %
      pg->cfg.size;

  int listen_port =
      pg->cfg.base_port +
      pg->cfg.rank;

  int next_port =
      pg->cfg.base_port +
      next_rank;

  fprintf(stderr,
          "Rank %d: listening on TCP port %d\n",
          pg->cfg.rank,
          listen_port);

  listener =
      create_listener(listen_port);

  if (listener < 0) {
    fprintf(stderr,
            "Rank %d: create_listener(%d) failed: %s\n",
            pg->cfg.rank,
            listen_port,
            strerror(errno));
    goto error;
  }

  fprintf(stderr,
          "Rank %d: connecting to rank %d at %s:%d\n",
          pg->cfg.rank,
          next_rank,
          pg->cfg.hosts[next_rank],
          next_port);

  pg->socket_to_next =
      connect_with_retry(
          pg->cfg.hosts[next_rank],
          next_port
      );

  if (pg->socket_to_next < 0) {
    fprintf(stderr,
            "Rank %d: failed connecting to %s:%d\n",
            pg->cfg.rank,
            pg->cfg.hosts[next_rank],
            next_port);
    goto error;
  }

  if (enable_tcp_nodelay(pg->socket_to_next) != 0) {
    goto error;
  }

  pg->socket_from_prev =
      accept(listener, NULL, NULL);

  if (pg->socket_from_prev < 0) {
    fprintf(stderr,
            "Rank %d: accept failed: %s\n",
            pg->cfg.rank,
            strerror(errno));
    goto error;
  }

  if (enable_tcp_nodelay(pg->socket_from_prev) != 0) {
    goto error;
  }

  close(listener);
  listener = -1;

  fprintf(stderr,
          "Rank %d: TCP ring connected\n",
          pg->cfg.rank);

  wire_mr_info_t local_control_wire = {
      .address = htobe64(
          (uint64_t)(uintptr_t)&pg->rendezvous_control),
      .rkey = htonl(pg->rendezvous_control_mr->rkey),
      .bytes = htonl((uint32_t)sizeof(pg->rendezvous_control))
  };

  wire_mr_info_t next_control_wire;
  memset(&next_control_wire, 0, sizeof(next_control_wire));

  if (send_all(pg->socket_from_prev,
               &local_control_wire,
               sizeof(local_control_wire)) != 0 ||
      recv_all(pg->socket_to_next,
               &next_control_wire,
               sizeof(next_control_wire)) != 0) {
    fprintf(stderr,
            "Rank %d: failed exchanging Rendezvous control MR\n",
            pg->cfg.rank);
    goto error;
  }

  pg->next_remote_control_mr.address =
      be64toh(next_control_wire.address);
  pg->next_remote_control_mr.rkey =
      ntohl(next_control_wire.rkey);
  pg->next_remote_control_mr.bytes =
      ntohl(next_control_wire.bytes);

  if (pg->next_remote_control_mr.bytes <
      sizeof(rendezvous_control_t)) {
    fprintf(stderr,
            "Rank %d: next Rendezvous control MR is too small\n",
            pg->cfg.rank);
    goto error;
  }

  /* -------------------------------------------------------------- */
  /* Exchange QP information                                       */
  /* -------------------------------------------------------------- */

  uint32_t next_local_psn =
      create_local_psn(pg->cfg.rank, 0);

  uint32_t prev_local_psn =
      create_local_psn(pg->cfg.rank, 1);

  if (exchange_qp_information(
      pg,
      next_local_psn,
      prev_local_psn) != 0) {
    goto error;
  }

  /* -------------------------------------------------------------- */
  /* Move QPs INIT -> RTR                                           */
  /* -------------------------------------------------------------- */

  /*
   * qp_next connects to the next rank's qp_prev.
   */
  if (modify_qp_to_rtr(
      pg,
      pg->qp_next,
      &pg->next_qp_info) != 0) {
    goto error;
  }

  /*
   * qp_prev connects to the previous rank's qp_next.
   */
  if (modify_qp_to_rtr(
      pg,
      pg->qp_prev,
      &pg->prev_qp_info) != 0) {
    goto error;
  }

  /* -------------------------------------------------------------- */
  /* Move QPs RTR -> RTS                                           */
  /* -------------------------------------------------------------- */

  if (modify_qp_to_rts(
      pg,
      pg->qp_next,
      next_local_psn) != 0) {
    goto error;
  }

  if (modify_qp_to_rts(
      pg,
      pg->qp_prev,
      prev_local_psn) != 0) {
    goto error;
  }

  fprintf(stderr,
          "Rank %d: both QPs are RTS\n",
          pg->cfg.rank);

  /*
   * Fill the receive queue once. Every consumed slot is reposted
   * immediately by the collective code, so the peer never observes
   * an empty receive queue.
   */
  for (int slot = 0; slot < PG_EAGER_RECV_SLOTS; ++slot) {
    if (post_eager_receive_slot(
        pg,
        slot,
        pg->eager_slot_size) != 0) {
      goto error;
    }
  }

  /*
   * Ensure every process has finished QP setup and receive posting
   * before the first collective. This setup barrier is not timed.
   */
  if (pg_barrier(pg) != 0) {
    fprintf(stderr,
            "Rank %d: initial barrier failed\n",
            pg->cfg.rank);
    goto error;
  }

  *pg_handle = pg;
  return 0;

  error:
  if (listener >= 0) {
    close(listener);
  }

  if (pg) {
    pg_release_work_buffer(pg);

    if (pg->socket_to_next >= 0) {
      close(pg->socket_to_next);
    }

    if (pg->socket_from_prev >= 0) {
      close(pg->socket_from_prev);
    }

    if (pg->qp_next) {
      ibv_destroy_qp(pg->qp_next);
    }

    if (pg->qp_prev) {
      ibv_destroy_qp(pg->qp_prev);
    }

    if (pg->rendezvous_control_mr) {
      ibv_dereg_mr(pg->rendezvous_control_mr);
      pg->rendezvous_control_mr = NULL;
    }

    if (pg->eager_recv_mr) {
      ibv_dereg_mr(pg->eager_recv_mr);
      pg->eager_recv_mr = NULL;
    }

    free(pg->eager_recv_pool);
    pg->eager_recv_pool = NULL;

    if (pg->cq) {
      ibv_destroy_cq(pg->cq);
    }

    if (pg->pd) {
      ibv_dealloc_pd(pg->pd);
    }

    if (pg->context) {
      ibv_close_device(pg->context);
    }

    if (pg->device_list) {
      ibv_free_device_list(
          pg->device_list
      );
    }

    free(pg);
  }

  return -1;
}

int pg_close(void *pg_handle)
{
  pg_handle_t *pg =
      (pg_handle_t *)pg_handle;

  if (!pg) {
    return 0;
  }

  /*
   * Stop RDMA operations before destroying the CQ and PD.
   */
  pg_release_work_buffer(pg);

  if (pg->qp_next) {
    ibv_destroy_qp(pg->qp_next);
    pg->qp_next = NULL;
  }

  if (pg->qp_prev) {
    ibv_destroy_qp(pg->qp_prev);
    pg->qp_prev = NULL;
  }

  if (pg->rendezvous_control_mr) {
    ibv_dereg_mr(pg->rendezvous_control_mr);
    pg->rendezvous_control_mr = NULL;
  }

  if (pg->eager_recv_mr) {
    ibv_dereg_mr(pg->eager_recv_mr);
    pg->eager_recv_mr = NULL;
  }

  free(pg->eager_recv_pool);
  pg->eager_recv_pool = NULL;

  if (pg->socket_to_next >= 0) {
    close(pg->socket_to_next);
    pg->socket_to_next = -1;
  }

  if (pg->socket_from_prev >= 0) {
    close(pg->socket_from_prev);
    pg->socket_from_prev = -1;
  }

  if (pg->cq) {
    ibv_destroy_cq(pg->cq);
    pg->cq = NULL;
  }

  if (pg->pd) {
    ibv_dealloc_pd(pg->pd);
    pg->pd = NULL;
  }

  if (pg->context) {
    ibv_close_device(pg->context);
    pg->context = NULL;
  }

  if (pg->device_list) {
    ibv_free_device_list(
        pg->device_list
    );

    pg->device_list = NULL;
  }

  free(pg);
  return 0;
}

static void format_message_size(size_t bytes,
                                char *output,
                                size_t output_size)
{
  if (bytes >= 1024U * 1024U) {
    snprintf(output, output_size, "%.0f MiB",
             (double)bytes / (1024.0 * 1024.0));
  }
  else if (bytes >= 1024U) {
    snprintf(output, output_size, "%.0f KiB",
             (double)bytes / 1024.0);
  }
  else {
    snprintf(output, output_size, "%zu B", bytes);
  }
}

int main(int argc, char **argv)
{
  memset(&g_config, 0, sizeof(g_config));

  g_config.rank = -1;
  g_config.size = 0;
  g_config.base_port = PG_DEFAULT_PORT;
  g_config.ib_port = 1;
  g_config.gid_index = 0;

  /*
   * The benchmark changes this field before testing each protocol.
   */
  g_config.protocol = PROTOCOL_EAGER;

  int iterations = 100;
  int warmup_iterations = 10;

  /*
   * Message sizes to compare.
   *
   * The current eager All-Gather requires:
   *
   *   chunk_bytes <= eager_slot_size
   *
   * With two processes and a 131072-byte Eager receive slot,
   * the maximum total message size is 262144 bytes.
   */
  static const size_t message_sizes_bytes[] = {
      256,
      512,
      1024,
      2048,
      4096,
      8192,
      16384,
      32768,
      65536,
      131072,
      262144
  };

  const size_t number_of_sizes =
      sizeof(message_sizes_bytes) /
      sizeof(message_sizes_bytes[0]);

  /* ------------------------------------------------------------------ */
  /* Parse command-line arguments                                       */
  /* ------------------------------------------------------------------ */

  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "-myindex") == 0) {
      if (i + 1 >= argc) {
        fprintf(stderr,
                "Error: -myindex requires a rank\n");
        return EXIT_FAILURE;
      }

      g_config.rank = parse_rank(argv[++i]);
    }

    else if (strcmp(argv[i], "-list") == 0) {
      while (i + 1 < argc &&
             argv[i + 1][0] != '-') {

        if (g_config.size >= PG_MAX_HOSTS) {
          fprintf(stderr,
                  "Error: too many hosts. "
                  "Maximum supported hosts: %d\n",
                  PG_MAX_HOSTS);
          return EXIT_FAILURE;
        }

        ++i;

        snprintf(
            g_config.hosts[g_config.size],
            PG_MAX_HOST_LEN,
            "%s",
            argv[i]
        );

        ++g_config.size;
      }
    }

    else if (strcmp(argv[i], "-iters") == 0) {
      if (i + 1 >= argc) {
        fprintf(stderr,
                "Error: -iters requires a value\n");
        return EXIT_FAILURE;
      }

      char *end = NULL;
      long value = strtol(argv[++i], &end, 10);

      if (*end != '\0' ||
          value <= 0 ||
          value > INT32_MAX) {

        fprintf(stderr,
                "Error: invalid iteration count '%s'\n",
                argv[i]);
        return EXIT_FAILURE;
      }

      iterations = (int)value;
    }

    else if (strcmp(argv[i], "-warmup") == 0) {
      if (i + 1 >= argc) {
        fprintf(stderr,
                "Error: -warmup requires a value\n");
        return EXIT_FAILURE;
      }

      char *end = NULL;
      long value = strtol(argv[++i], &end, 10);

      if (*end != '\0' ||
          value < 0 ||
          value > INT32_MAX) {

        fprintf(stderr,
                "Error: invalid warm-up count '%s'\n",
                argv[i]);
        return EXIT_FAILURE;
      }

      warmup_iterations = (int)value;
    }

    else {
      fprintf(stderr,
              "Error: unknown argument '%s'\n",
              argv[i]);

      fprintf(stderr,
              "Usage:\n"
              "  %s "
              "-myindex <01|02|...> "
              "-list <host1> <host2> [...] "
              "[-iters N] "
              "[-warmup N]\n",
              argv[0]);

      return EXIT_FAILURE;
    }
  }

  /* ------------------------------------------------------------------ */
  /* Validate configuration                                             */
  /* ------------------------------------------------------------------ */

  if (g_config.size < 2) {
    fprintf(stderr,
            "Error: at least two hosts are required\n");
    return EXIT_FAILURE;
  }

  if (g_config.rank < 0 ||
      g_config.rank >= g_config.size) {

    fprintf(stderr,
            "Error: rank must be between 0 and %d\n",
            g_config.size - 1);
    return EXIT_FAILURE;
  }

  printf(
      "Rank %d starting protocol comparison:\n"
      "  Processes:             %d\n"
      "  Message sizes:         %zu\n"
      "  Warm-up per protocol:  %d\n"
      "  Iterations per test:   %d\n",
      g_config.rank,
      g_config.size,
      number_of_sizes,
      warmup_iterations,
      iterations
  );

  fflush(stdout);

  /* ------------------------------------------------------------------ */
  /* Connect process group once                                         */
  /* ------------------------------------------------------------------ */

  void *pg_handle = NULL;

  if (connect_process_group(NULL, &pg_handle) != 0) {
    fprintf(stderr,
            "Rank %d: failed to connect process group\n",
            g_config.rank);
    return EXIT_FAILURE;
  }

  pg_handle_t *pg =
      (pg_handle_t *)pg_handle;

  /*
   * Store results for printing the final comparison table.
   */
  double *eager_times_us =
      calloc(number_of_sizes, sizeof(*eager_times_us));

  double *rendezvous_times_us =
      calloc(number_of_sizes, sizeof(*rendezvous_times_us));

  double *eager_bandwidth_gbps =
      calloc(number_of_sizes, sizeof(*eager_bandwidth_gbps));

  double *rendezvous_bandwidth_gbps =
      calloc(number_of_sizes, sizeof(*rendezvous_bandwidth_gbps));

  bool *size_was_tested =
      calloc(number_of_sizes, sizeof(*size_was_tested));

  if (!eager_times_us ||
      !rendezvous_times_us ||
      !eager_bandwidth_gbps ||
      !rendezvous_bandwidth_gbps ||
      !size_was_tested) {

    perror("calloc");

    free(eager_times_us);
    free(rendezvous_times_us);
    free(eager_bandwidth_gbps);
    free(rendezvous_bandwidth_gbps);
    free(size_was_tested);

    pg_close(pg_handle);
    return EXIT_FAILURE;
  }

  const PROTOCOL_MODE protocols[] = {
      PROTOCOL_EAGER,
      PROTOCOL_RENDEZVOUS
  };

  const size_t number_of_protocols =
      sizeof(protocols) /
      sizeof(protocols[0]);

  int32_t expected =
      (int32_t)(
          g_config.size *
          (g_config.size + 1) / 2
      );

  /* ------------------------------------------------------------------ */
  /* Benchmark all message sizes                                        */
  /* ------------------------------------------------------------------ */

  for (size_t size_index = 0;
       size_index < number_of_sizes;
       ++size_index) {

    size_t message_bytes =
        message_sizes_bytes[size_index];

    if (message_bytes % sizeof(int32_t) != 0) {
      if (g_config.rank == 0) {
        fprintf(stderr,
                "Skipping %zu bytes: not divisible "
                "by int32_t size\n",
                message_bytes);
      }

      continue;
    }

    size_t count_size_t =
        message_bytes / sizeof(int32_t);

    if (count_size_t > INT32_MAX) {
      if (g_config.rank == 0) {
        fprintf(stderr,
                "Skipping %zu bytes: element count "
                "is too large\n",
                message_bytes);
      }

      continue;
    }

    int count =
        (int)count_size_t;

    /*
     * Ring Reduce-Scatter divides the buffer into P chunks.
     */
    if (count % g_config.size != 0) {
      if (g_config.rank == 0) {
        fprintf(stderr,
                "Skipping %zu bytes: element count %d "
                "is not divisible by %d processes\n",
                message_bytes,
                count,
                g_config.size);
      }

      continue;
    }

    size_t chunk_bytes =
        message_bytes /
        (size_t)g_config.size;

  /*
   * The Eager All-Gather receives an entire chunk into one
   * pre-posted Eager receive slot.
   */
    if (chunk_bytes > pg->eager_slot_size) {
      if (g_config.rank == 0) {
        fprintf(stderr,
                "Skipping %zu bytes: chunk size %zu "
                "exceeds Eager receive slot size %zu\n",
                message_bytes,
                chunk_bytes,
                pg->eager_slot_size);
      }

      continue;
    }

    int32_t *sendbuf =
        malloc(message_bytes);

    int32_t *recvbuf =
        malloc(message_bytes);

    if (!sendbuf || !recvbuf) {
      perror("malloc");

      pg_release_work_buffer(pg);
      free(sendbuf);
      free(recvbuf);

      free(eager_times_us);
      free(rendezvous_times_us);
      free(eager_bandwidth_gbps);
      free(rendezvous_bandwidth_gbps);
      free(size_was_tested);

      pg_close(pg_handle);
      return EXIT_FAILURE;
    }

    int32_t local_value =
        (int32_t)(g_config.rank + 1);

    for (int i = 0; i < count; ++i) {
      sendbuf[i] = local_value;
      recvbuf[i] = 0;
    }

    /* Register and exchange MR metadata once per size, outside timing. */
    if (pg_prepare_work_buffer(pg, recvbuf, message_bytes) != 0) {
      fprintf(stderr,
              "Rank %d: failed preparing %zu-byte work buffer\n",
              g_config.rank,
              message_bytes);

      pg_release_work_buffer(pg);
      free(sendbuf);
      free(recvbuf);
      free(eager_times_us);
      free(rendezvous_times_us);
      free(eager_bandwidth_gbps);
      free(rendezvous_bandwidth_gbps);
      free(size_was_tested);
      pg_close(pg_handle);
      return EXIT_FAILURE;
    }

    if (g_config.rank == 0) {
      char size_text[32];
      format_message_size(message_bytes, size_text, sizeof(size_text));

      printf(
          "\nTesting message size: %s (%zu bytes, %d int32 elements)\n",
          size_text,
          message_bytes,
          count
      );

      fflush(stdout);
    }

    for (size_t protocol_index = 0;
         protocol_index < number_of_protocols;
         ++protocol_index) {

      PROTOCOL_MODE protocol =
          protocols[protocol_index];

      pg->cfg.protocol = protocol;

      const char *protocol_name =
          protocol == PROTOCOL_EAGER
          ? "EAGER"
          : "RENDEZVOUS";

      /*
       * Synchronize before beginning this protocol test.
       */
      if (pg_barrier(pg) != 0) {
        fprintf(stderr,
                "Rank %d: barrier failed before %s test\n",
                g_config.rank,
                protocol_name);

        pg_release_work_buffer(pg);
        free(sendbuf);
        free(recvbuf);

        free(eager_times_us);
        free(rendezvous_times_us);
        free(eager_bandwidth_gbps);
        free(rendezvous_bandwidth_gbps);
        free(size_was_tested);

        pg_close(pg_handle);
        return EXIT_FAILURE;
      }

      /* -------------------------------------------------------------- */
      /* Warm-up                                                        */
      /* -------------------------------------------------------------- */

      for (int iteration = 0;
           iteration < warmup_iterations;
           ++iteration) {

        /*
         * Prevent ranks from drifting between warm-up iterations.
         */
        if (pg_barrier(pg) != 0) {
          fprintf(stderr,
                  "Rank %d: barrier failed before %s warm-up "
                  "iteration %d\n",
                  g_config.rank,
                  protocol_name,
                  iteration);

          pg_release_work_buffer(pg);
          free(sendbuf);
          free(recvbuf);

          free(eager_times_us);
          free(rendezvous_times_us);
          free(eager_bandwidth_gbps);
          free(rendezvous_bandwidth_gbps);
          free(size_was_tested);

          pg_close(pg_handle);
          return EXIT_FAILURE;
        }

        if (pg_all_reduce(
            sendbuf,
            recvbuf,
            count,
            PG_INT32,
            PG_SUM,
            pg) != 0) {

          fprintf(stderr,
                  "Rank %d: %s warm-up failed "
                  "for %zu bytes at iteration %d\n",
                  g_config.rank,
                  protocol_name,
                  message_bytes,
                  iteration);

          pg_release_work_buffer(pg);
          free(sendbuf);
          free(recvbuf);

          free(eager_times_us);
          free(rendezvous_times_us);
          free(eager_bandwidth_gbps);
          free(rendezvous_bandwidth_gbps);
          free(size_was_tested);

          pg_close(pg_handle);
          return EXIT_FAILURE;
        }
      }

      /*
       * Verify the warm-up result.
       */
      for (int i = 0; i < count; ++i) {
        if (recvbuf[i] != expected) {
          fprintf(stderr,
                  "Rank %d: %s correctness failure "
                  "for %zu bytes at index %d: "
                  "got %d, expected %d\n",
                  g_config.rank,
                  protocol_name,
                  message_bytes,
                  i,
                  recvbuf[i],
                  expected);

          pg_release_work_buffer(pg);
          free(sendbuf);
          free(recvbuf);

          free(eager_times_us);
          free(rendezvous_times_us);
          free(eager_bandwidth_gbps);
          free(rendezvous_bandwidth_gbps);
          free(size_was_tested);

          pg_close(pg_handle);
          return EXIT_FAILURE;
        }
      }

      double total_elapsed_seconds = 0.0;

      /* -------------------------------------------------------------- */
      /* Timed iterations                                               */
      /* -------------------------------------------------------------- */

      for (int iteration = 0;
           iteration < iterations;
           ++iteration) {

        /*
         * Synchronize before this individual collective.
         *
         * This barrier is outside the measured interval.
         */
        if (pg_barrier(pg) != 0) {
          fprintf(stderr,
                  "Rank %d: barrier failed before %s "
                  "iteration %d\n",
                  g_config.rank,
                  protocol_name,
                  iteration);

          pg_release_work_buffer(pg);
          free(sendbuf);
          free(recvbuf);

          free(eager_times_us);
          free(rendezvous_times_us);
          free(eager_bandwidth_gbps);
          free(rendezvous_bandwidth_gbps);
          free(size_was_tested);

          pg_close(pg_handle);
          return EXIT_FAILURE;
        }

        struct timespec iteration_start;
        struct timespec iteration_end;

        if (clock_gettime(
            CLOCK_MONOTONIC,
            &iteration_start) != 0) {

          perror("clock_gettime");

          pg_release_work_buffer(pg);
          free(sendbuf);
          free(recvbuf);

          free(eager_times_us);
          free(rendezvous_times_us);
          free(eager_bandwidth_gbps);
          free(rendezvous_bandwidth_gbps);
          free(size_was_tested);

          pg_close(pg_handle);
          return EXIT_FAILURE;
        }

        if (pg_all_reduce(
            sendbuf,
            recvbuf,
            count,
            PG_INT32,
            PG_SUM,
            pg) != 0) {

          fprintf(stderr,
                  "Rank %d: %s benchmark failed "
                  "for %zu bytes at iteration %d\n",
                  g_config.rank,
                  protocol_name,
                  message_bytes,
                  iteration);

          pg_release_work_buffer(pg);
          free(sendbuf);
          free(recvbuf);

          free(eager_times_us);
          free(rendezvous_times_us);
          free(eager_bandwidth_gbps);
          free(rendezvous_bandwidth_gbps);
          free(size_was_tested);

          pg_close(pg_handle);
          return EXIT_FAILURE;
        }

        if (clock_gettime(
            CLOCK_MONOTONIC,
            &iteration_end) != 0) {

          perror("clock_gettime");

          pg_release_work_buffer(pg);
          free(sendbuf);
          free(recvbuf);

          free(eager_times_us);
          free(rendezvous_times_us);
          free(eager_bandwidth_gbps);
          free(rendezvous_bandwidth_gbps);
          free(size_was_tested);

          pg_close(pg_handle);
          return EXIT_FAILURE;
        }

        double iteration_seconds =
            (double)(
                iteration_end.tv_sec -
                iteration_start.tv_sec
            );

        iteration_seconds +=
            (double)(
                iteration_end.tv_nsec -
                iteration_start.tv_nsec
            ) / 1e9;

        total_elapsed_seconds +=
            iteration_seconds;
      }


      /*
 * Synchronize after timing, but do not include the barrier
 * in the measured protocol latency.
 */
      if (pg_barrier(pg) != 0) {
        fprintf(stderr,
                "Rank %d: barrier failed after timing %s\n",
                g_config.rank,
                protocol_name);

        pg_release_work_buffer(pg);
        free(sendbuf);
        free(recvbuf);

        free(eager_times_us);
        free(rendezvous_times_us);
        free(eager_bandwidth_gbps);
        free(rendezvous_bandwidth_gbps);
        free(size_was_tested);

        pg_close(pg_handle);
        return EXIT_FAILURE;
      }

      /*
       * Verify the final timed result.
       */
      for (int i = 0; i < count; ++i) {
        if (recvbuf[i] != expected) {
          fprintf(stderr,
                  "Rank %d: final %s correctness failure "
                  "for %zu bytes at index %d: "
                  "got %d, expected %d\n",
                  g_config.rank,
                  protocol_name,
                  message_bytes,
                  i,
                  recvbuf[i],
                  expected);

          pg_release_work_buffer(pg);
          free(sendbuf);
          free(recvbuf);

          free(eager_times_us);
          free(rendezvous_times_us);
          free(eager_bandwidth_gbps);
          free(rendezvous_bandwidth_gbps);
          free(size_was_tested);

          pg_close(pg_handle);
          return EXIT_FAILURE;
        }
      }

      double average_seconds =
          total_elapsed_seconds /
          (double)iterations;

      double average_microseconds =
          average_seconds * 1e6;

      /*
       * Ring traffic per rank:
       *
       * 2(P-1)/P × message size
       */
      double ring_traffic_bytes =
          2.0 *
          ((double)(g_config.size - 1) /
           (double)g_config.size) *
          (double)message_bytes;

      double algorithmic_gbps =
          (ring_traffic_bytes * 8.0) /
          average_seconds /
          1e9;

      if (protocol == PROTOCOL_EAGER) {
        eager_times_us[size_index] =
            average_microseconds;

        eager_bandwidth_gbps[size_index] =
            algorithmic_gbps;
      }
      else {
        rendezvous_times_us[size_index] =
            average_microseconds;

        rendezvous_bandwidth_gbps[size_index] =
            algorithmic_gbps;
      }

      size_was_tested[size_index] = true;

      if (g_config.rank == 0) {
        printf(
            "  %-12s: %10.2f us, %10.3f Gbps\n",
            protocol_name,
            average_microseconds,
            algorithmic_gbps
        );

        fflush(stdout);
      }
    }

    pg_release_work_buffer(pg);
    free(sendbuf);
    free(recvbuf);
  }

  /* ------------------------------------------------------------------ */
  /* Print final comparison table                                       */
  /* ------------------------------------------------------------------ */

  if (g_config.rank == 0) {
    printf(
        "\n"
        "Protocol comparison\n"
        "=================================================================================================\n"
        "%12s | %12s | %12s | %12s | %12s | %12s\n"
        "-------------------------------------------------------------------------------------------------\n",
        "Message Size",
        "Eager us",
        "Rendezvous",
        "Eager Gbps",
        "Rendez Gbps",
        "Winner"
    );

    for (size_t size_index = 0;
         size_index < number_of_sizes;
         ++size_index) {

      if (!size_was_tested[size_index]) {
        continue;
      }

      char size_text[32];

      format_message_size(
          message_sizes_bytes[size_index],
          size_text,
          sizeof(size_text)
      );

      const char *winner;

      if (eager_times_us[size_index] <
          rendezvous_times_us[size_index]) {

        winner = "EAGER";
      }
      else if (rendezvous_times_us[size_index] <
               eager_times_us[size_index]) {

        winner = "RENDEZVOUS";

      }
      else {
        winner = "TIE";
      }

      printf(
          "%12s | %12.2f | %12.2f | %12.3f | %12.3f | %12s\n",
          size_text,
          eager_times_us[size_index],
          rendezvous_times_us[size_index],
          eager_bandwidth_gbps[size_index],
          rendezvous_bandwidth_gbps[size_index],
          winner
      );
    }

    printf(
        "=================================================================================================\n"
    );

    size_t stable_threshold = 0;

    for (size_t i = 0; i + 2 < number_of_sizes; ++i) {
      if (!size_was_tested[i] ||
          !size_was_tested[i + 1] ||
          !size_was_tested[i + 2]) {
        continue;
      }

      bool first_win =
          rendezvous_times_us[i] <= 0.95 * eager_times_us[i];
      bool second_win =
          rendezvous_times_us[i + 1] <= 0.95 * eager_times_us[i + 1];
      bool third_win =
          rendezvous_times_us[i + 2] <= 0.95 * eager_times_us[i + 2];

      if (first_win && second_win && third_win) {
        stable_threshold = message_sizes_bytes[i];
        break;
      }
    }

    if (stable_threshold != 0) {
      char threshold_text[32];
      format_message_size(stable_threshold,
                          threshold_text,
                          sizeof(threshold_text));

      printf(
          "\nRecommended protocol crossover based on benchmark results: %s "
          "(%zu bytes)\n"
          "Rendezvous was at least 5%% faster for three "
          "consecutive sizes.\n",
          threshold_text,
          stable_threshold
      );
    }
    else {
      printf(
          "\nNo stable protocol threshold was found.\n"
          "A threshold requires Rendezvous to be at least 5%% faster "
          "for three consecutive sizes.\n"
      );
    }

    fflush(stdout);
  }

  /* ------------------------------------------------------------------ */
  /* Cleanup                                                            */
  /* ------------------------------------------------------------------ */

  free(eager_times_us);
  free(rendezvous_times_us);
  free(eager_bandwidth_gbps);
  free(rendezvous_bandwidth_gbps);
  free(size_was_tested);

  if (pg_close(pg_handle) != 0) {
    fprintf(stderr,
            "Rank %d: process-group cleanup failed\n",
            g_config.rank);
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}