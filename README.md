# RDMA Ring All-Reduce: Eager vs. Rendezvous

This project implements and benchmarks a **Ring All-Reduce collective using the Linux RDMA Verbs API**.  
The implementation performs All-Reduce as two ring phases:

1. **Reduce-Scatter** – processes exchange chunks around the ring while applying a local reduction.
2. **All-Gather** – the reduced chunks are circulated until every process holds the complete result.

The benchmark compares two communication protocols, **Eager** and **Rendezvous**, over several message sizes and reports average latency, algorithmic bandwidth, and a possible stable crossover threshold.

## Main Features

- Ring-based **Reduce-Scatter + All-Gather**
- Linux **RDMA Verbs (`libibverbs`)**
- Reliable Connected (**RC**) Queue Pairs
- Support for **2–4 processes**
- **Eager protocol** using RDMA SEND/RECV
- **Rendezvous protocol** using one-sided RDMA WRITE
- Pre-posted Eager receive buffers
- Registered-memory reuse outside the timed hot path
- Micro-chunk pipelining during Reduce-Scatter
- TCP-based bootstrap, metadata exchange, and synchronization
- Support in the collective API for:
  - `int32`
  - `float`
  - `double`
- Reduction operations:
  - `SUM`
  - `MAX`
  - `MIN`
- Automatic benchmark comparison over message sizes from **256 B to 256 KiB**
- Correctness verification after warm-up and timed execution
- Automatic detection of a possible stable Eager/Rendezvous threshold

## Communication Design

Each process is assigned a rank and connected in a logical ring:

```text
Rank 0 -> Rank 1 -> ... -> Rank P-1 -> Rank 0
```

Every process creates two RC Queue Pairs:

- `qp_next` – sends to the next rank
- `qp_prev` – receives from the previous rank

TCP is used only for bootstrap/control operations such as establishing the ring, exchanging QP and memory-region information, and barriers. The collective data path itself uses RDMA.

### Reduce-Scatter

The input vector is divided into `P` equal chunks, where `P` is the number of processes.

During each of the `P - 1` steps, every rank:

1. sends one chunk to its next neighbor,
2. receives a chunk from its previous neighbor,
3. reduces the received values into its local chunk.

The Reduce-Scatter implementation divides chunks into smaller micro-chunks (`PG_MICRO_CHUNK_SIZE = 4096` elements) so communication and local reduction can be performed incrementally.

### All-Gather

After Reduce-Scatter, each process owns one fully reduced chunk.  
The All-Gather phase circulates these chunks for another `P - 1` steps until every process has the complete reduced vector.

## Eager Protocol

The Eager path uses **RDMA SEND/RECV**.

Receive buffers are registered in advance and **64 receive slots are pre-posted**. When a receive completes, its slot is consumed and immediately reposted, keeping the receive queue populated.

This avoids registering memory or creating receive buffers inside the timed communication path.

## Rendezvous Protocol

The Rendezvous All-Gather uses **one-sided RDMA WRITE**.

Before benchmarking, neighboring processes exchange the address and `rkey` of their registered work buffers. A sender can therefore write a chunk directly into the correct location in the next process's buffer.

For synchronization, each data write is followed by an RDMA WRITE of an **epoch flag**. RC ordering guarantees that the data becomes visible before the corresponding flag. The receiver waits for that epoch value before continuing.

## Building

### Requirements

The project requires:

- Linux
- GCC with C11 support
- RDMA-capable machines
- InfiniBand or RoCE
- `libibverbs` development libraries

Build with:

```bash
make
```

Equivalent command:

```bash
gcc -std=c11 -O2 -Wall -Wextra -Wpedantic \
    ring_allreduce_advanced.c -o ring_allreduce -libverbs
```

Clean generated files with:

```bash
make clean
```

> **Source filename:** the supplied Makefile expects the implementation to be named `ring_allreduce_advanced.c`. If your source file currently has another name, either rename it or update the `SRC` variable in the Makefile.

## Running

The program must be started on **all participating machines** with the same ordered host list and a different `-myindex`.

General syntax:

```bash
./ring_allreduce \
    -myindex <rank> \
    -list <host1> <host2> [...] \
    [-iters N] \
    [-warmup N]
```

`-myindex` accepts values such as `01`, `02`, etc. The program converts these to zero-based internal ranks.

### Example: 2 Processes

On `mlx-stud-01`:

```bash
./ring_allreduce \
    -myindex 01 \
    -list mlx-stud-01 mlx-stud-02 \
    -iters 1000 \
    -warmup 50
```

On `mlx-stud-02`:

```bash
./ring_allreduce \
    -myindex 02 \
    -list mlx-stud-01 mlx-stud-02 \
    -iters 1000 \
    -warmup 50
```

### Example: 4 Processes

Run the program on all four machines using the same list:

```bash
./ring_allreduce \
    -myindex 01 \
    -list mlx-stud-01 mlx-stud-02 mlx-stud-03 mlx-stud-04 \
    -iters 1000 \
    -warmup 50
```

Change `-myindex` to `02`, `03`, and `04` on the other three machines.

## Benchmark Methodology

The benchmark evaluates these message sizes:

```text
256 B
512 B
1 KiB
2 KiB
4 KiB
8 KiB
16 KiB
32 KiB
64 KiB
128 KiB
256 KiB
```

For each size, the program benchmarks both Eager and Rendezvous.

The supplied experiments use:

- **50 warm-up iterations**
- **1000 timed iterations**
- `int32_t` elements
- `SUM` reduction

Each rank initializes every element to:

```text
rank + 1
```

Therefore, the expected value after All-Reduce is:

```text
P(P + 1) / 2
```

where `P` is the number of processes. The implementation checks the output for correctness.

A barrier is performed before each timed collective, but the barrier itself is **outside the measured interval**.

## Performance Metrics

The reported latency is the average execution time of `pg_all_reduce()`.

Algorithmic bandwidth is calculated from the amount of ring traffic handled by each rank:

```text
Ring traffic per rank = 2(P - 1) / P * message_size
```

and:

```text
Bandwidth = ring_traffic_bits / average_time
```

The final table selects the protocol with the lower measured latency for each message size.

## Experimental Results

### 2 Processes

| Message Size | Eager (µs) | Rendezvous (µs) | Eager (Gbps) | Rendezvous (Gbps) | Winner |
|---:|---:|---:|---:|---:|:---|
| 256 B | 11.01 | 11.22 | 0.185929 | 0.182454 | Eager |
| 512 B | 11.03 | 11.20 | 0.371425 | 0.365817 | Eager |
| 1 KiB | 12.31 | 14.28 | 0.665660 | 0.573713 | Eager |
| 2 KiB | 13.72 | 13.13 | 1.194239 | 1.248257 | Rendezvous |
| 4 KiB | 15.33 | 16.88 | 2.137341 | 1.941116 | Eager |
| 8 KiB | 18.35 | 18.70 | 3.571986 | 3.503808 | Eager |
| 16 KiB | 24.64 | 23.91 | 5.319599 | 5.481244 | Rendezvous |
| 32 KiB | 36.91 | 34.83 | 7.101401 | 7.526539 | Rendezvous |
| 64 KiB | 68.19 | 64.41 | 7.688407 | 8.139883 | Rendezvous |
| 128 KiB | 127.34 | 120.74 | 8.234494 | 8.684237 | Rendezvous |
| 256 KiB | 233.75 | 225.17 | 8.971963 | 9.313648 | Rendezvous |

The program reported a possible stable threshold of:

```text
32 KiB (32768 bytes)
```

At that point, Rendezvous was at least **5% faster for three consecutive tested sizes**.

### 4 Processes

| Message Size | Eager (µs) | Rendezvous (µs) | Eager (Gbps) | Rendezvous (Gbps) | Winner |
|---:|---:|---:|---:|---:|:---|
| 256 B | 27.56 | 30.52 | 0.111455 | 0.100663 | Eager |
| 512 B | 27.26 | 28.76 | 0.225425 | 0.213600 | Eager |
| 1 KiB | 30.45 | 33.21 | 0.403535 | 0.369969 | Eager |
| 2 KiB | 30.98 | 34.07 | 0.793278 | 0.721239 | Eager |
| 4 KiB | 36.92 | 37.43 | 1.331356 | 1.313147 | Eager |
| 8 KiB | 40.22 | 38.50 | 2.443988 | 2.553183 | Rendezvous |
| 16 KiB | 53.17 | 50.20 | 3.697691 | 3.916228 | Rendezvous |
| 32 KiB | 75.43 | 71.44 | 5.212725 | 5.503938 | Rendezvous |
| 64 KiB | 107.26 | 98.59 | 7.331954 | 7.976583 | Rendezvous |
| 128 KiB | 186.46 | 178.73 | 8.435408 | 8.800334 | Rendezvous |
| 256 KiB | 346.51 | 329.15 | 9.078273 | 9.557001 | Rendezvous |

The program reported a possible stable threshold of:

```text
16 KiB (16384 bytes)
```

using the same 5%-for-three-consecutive-sizes criterion.

## Results Discussion

The measurements show the expected trade-off between the two approaches.

For **small messages**, Eager generally performs better because SEND/RECV has low overhead and the data is immediately transferred through already-posted receive buffers.

As message size increases, **Rendezvous becomes more competitive and then consistently faster**. Its one-sided RDMA WRITE path places data directly into the destination work buffer and avoids the extra receive-buffer copy used by the Eager All-Gather.

The crossover also changes with process count:

- **2 processes:** stable threshold reported at **32 KiB**
- **4 processes:** stable threshold reported at **16 KiB**

In these measurements, increasing the number of processes causes Rendezvous to become advantageous at a smaller message size.

## Stable Threshold Detection

The benchmark does not define the threshold as the first individual Rendezvous win.

Instead, a threshold is reported only when Rendezvous latency is at least **5% lower than Eager latency for three consecutive message sizes**:

```text
rendezvous_time <= 0.95 * eager_time
```

This reduces the chance of selecting a crossover point based on a single noisy measurement.

## Public API

The implementation exposes three main functions:

```c
int connect_process_group(char *servername, void **pg_handle);

int pg_all_reduce(
    void *sendbuf,
    void *recvbuf,
    int count,
    DATATYPE datatype,
    OPERATION op,
    void *pg_handle
);

int pg_close(void *pg_handle);
```

`connect_process_group()` initializes the TCP and RDMA resources, `pg_all_reduce()` performs the collective, and `pg_close()` releases the process-group resources.

## Implementation Constants

Important constants in the current implementation include:

```c
#define PG_DEFAULT_PORT       18515
#define PG_MAX_HOSTS          4
#define PG_CQ_SIZE            4096
#define PG_MAX_WR             512
#define PG_PIPELINE_STAGES    2
#define PG_MICRO_CHUNK_SIZE   4096
#define PG_EAGER_RECV_SLOTS   64
```

The QPs are configured as Reliable Connected (`IBV_QPT_RC`) and use an RDMA MTU of `IBV_MTU_1024` in the current implementation.

## Notes and Limitations

- The implementation currently supports at most **4 processes**.
- The element count must be divisible by the number of processes because the ring partitions the vector into equal chunks.
- The current benchmark tests `PG_INT32` with `PG_SUM`, although the collective implementation also contains `float`, `double`, `MAX`, and `MIN` support.
- The current Eager All-Gather requires a complete per-rank chunk to fit in the configured receive/pipeline buffer.
- TCP is part of setup and synchronization; it is not used to transfer collective payload data.
- The current benchmark automatically compares both protocols; the command-line parser does not require a `-protocol` option.

## Files

```text
.
├── ring_allreduce_advanced.c   # RDMA Ring All-Reduce implementation and benchmark
├── Makefile                    # Build rules
└── README.md                   # Project documentation
```

## Summary

This project demonstrates a complete RDMA Ring All-Reduce implementation and experimentally compares two communication strategies. The results illustrate why a hybrid collective can benefit from using **Eager for small messages** and switching to **Rendezvous for larger messages**, with the crossover depending on the number of participating processes.
