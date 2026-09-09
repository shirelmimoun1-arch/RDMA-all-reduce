# RDMA Ring All-Reduce: Eager vs. Rendezvous

A Linux RDMA Verbs implementation of **Ring All-Reduce** that compares two communication protocols in the **All-Gather** phase:

- **Eager** — two-sided `SEND/RECV` with pre-posted receive buffers.
- **Rendezvous** — one-sided `RDMA WRITE` followed by an ordered remote flag write.

The implementation uses a standard ring decomposition:

1. **Reduce-Scatter** — data is partitioned into chunks and reduced while circulating around the ring.
2. **All-Gather** — the reduced chunks circulate again until every rank owns the complete reduced vector.

The project also implements **micro-chunk pipelining**, **registered-memory reuse**, correctness checks, warm-up iterations, and an automatic benchmark-based protocol crossover recommendation.

---

## Highlights

- Linux **RDMA Verbs (`libibverbs`)**
- Reliable Connected (**RC**) Queue Pairs
- Ring topology with one logical predecessor and successor per rank
- **Pipelined Reduce-Scatter**
- **Eager vs. Rendezvous** All-Gather comparison
- Pre-posted Eager receive slots
- One-sided RDMA writes for Rendezvous
- Reuse of registered work buffers outside the timed hot path
- Support for native **InfiniBand** and **RoCE** addressing
- `PG_SUM`, `PG_MAX`, and `PG_MIN` local reduction operations
- `int32`, `float`, and `double` datatype support in the collective API
- Correctness validation during benchmarking
- Latency and algorithmic bandwidth measurements
- Automatic stable crossover detection

---

## Algorithm Overview

For `P` processes, each input vector is divided into `P` equal chunks.

### 1. Reduce-Scatter

Each rank initially copies its local input into its work buffer. The Reduce-Scatter phase then performs `P - 1` ring steps.

At each step, a rank:

- sends one chunk to its next neighbor,
- receives a chunk from its previous neighbor,
- performs the requested local reduction on the received data.

The implementation further divides a chunk into **micro-chunks**. Communication for the next micro-chunk is posted before reducing the current one, allowing network transfer and CPU reduction to overlap.

After `P - 1` steps, each rank owns one fully reduced chunk.

### 2. All-Gather

The reduced chunks then circulate for another `P - 1` steps. At the end, every rank has the complete All-Reduce result.

The benchmark executes this phase using both Eager and Rendezvous communication and compares their performance.

For the complete collective, the ring performs:

```text
(P - 1) Reduce-Scatter steps
+
(P - 1) All-Gather steps
=
2(P - 1) communication steps
```

---

## Communication Protocols

### Eager

The Eager path uses two-sided RDMA `SEND/RECV`.

Receive buffers are registered and **pre-posted** in a receive-slot pool. Each receive has its own slot, preventing an early message from overwriting data that has not yet been consumed.

A shared Completion Queue is polled for both send and receive completions. The implementation tracks completions that arrive earlier than the current operation expects.

**Why it works well for small messages:** the data can be sent immediately because receive buffers are already posted, so the protocol has relatively little synchronization overhead.

### Rendezvous

The Rendezvous All-Gather uses **one-sided `RDMA WRITE`**.

For each step, two Work Requests are chained:

```text
RDMA WRITE(data)  ->  RDMA WRITE(epoch flag)
```

The data write is unsignaled, while the flag write is signaled. Both are posted on the same RC QP. The receiver polls its local flag and treats the data as ready when the expected epoch becomes visible.

The epoch value changes on every All-Reduce invocation, so flags do not need to be explicitly cleared between operations.

**Why it becomes attractive for larger messages:** the payload is written directly into its final location in the remote registered work buffer, avoiding the Eager All-Gather's receive-slot-to-destination copy.

---

## Architecture

```text
                    TCP bootstrap / synchronization
             +-----------------------------------------+
             |                                         |
             v                                         v

       +-----------+      RC QP       +-----------+
       |  Rank i   | ----------------> | Rank i+1  |
       |           |                   |           |
       | qp_next   |                   | qp_prev   |
       +-----------+                   +-----------+
             ^                              |
             |                              |
             +--------- ring traffic -------+

RDMA data path:
  Reduce-Scatter : pipelined SEND/RECV + local reduction
  All-Gather     : Eager SEND/RECV  OR  Rendezvous RDMA WRITE
```

TCP is used for bootstrap metadata exchange and benchmark barriers; the collective data path itself uses RDMA.

---

## RDMA Setup

During process-group creation, each rank:

1. opens an RDMA device and queries the active port,
2. allocates a Protection Domain,
3. creates a Completion Queue,
4. creates two RC Queue Pairs:
   - `qp_next` for traffic toward the next rank,
   - `qp_prev` for traffic from the previous rank,
5. exchanges QP information over TCP,
6. moves QPs through `INIT -> RTR -> RTS`,
7. allocates and registers the Eager receive pool,
8. registers Rendezvous control flags and exchanges their remote memory information.

For native InfiniBand the connection uses **LID routing**. The code also contains GID selection logic for Ethernet/RoCE links.

---

## Public Collective API

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

Supported datatypes:

```text
PG_INT32
PG_FLOAT
PG_DOUBLE
```

Supported reduction operations:

```text
PG_SUM
PG_MAX
PG_MIN
```

The element count must be divisible by the number of participating processes because the ring implementation divides the vector into equal chunks.

---

## Requirements

- Linux
- RDMA-capable hosts
- Active InfiniBand or RoCE interface
- GCC with C11 support
- `libibverbs` development package
- TCP connectivity between participating hosts

The current implementation supports up to **4 processes**.

---

## Build

The provided Makefile builds the executable `ring_allreduce`:

```bash
make
```

Equivalent compilation command:

```bash
gcc -std=c11 -O2 -Wall -Wextra -Wpedantic \
    -D_POSIX_C_SOURCE=200809L \
    ring_allreduce.c -o ring_allreduce -libverbs
```

Clean generated files with:

```bash
make clean
```

> The Makefile expects the source file to be named `ring_allreduce.c`.

---

## Running the Benchmark

Run the program on **every participating host** with the same ordered host list. Only `-myindex` changes between hosts.

### Two processes

On `mlx-stud-01`:

```bash
./ring_allreduce \
    -myindex 01 \
    -list mlx-stud-01 mlx-stud-02 \
    -iters 3000 \
    -warmup 200
```

On `mlx-stud-02`, use the same command with:

```bash
-myindex 02
```

### Four processes

Use the same host list on all four machines:

```bash
./ring_allreduce \
    -myindex 01 \
    -list mlx-stud-01 mlx-stud-02 mlx-stud-03 mlx-stud-04 \
    -iters 3000 \
    -warmup 200
```

Then use `-myindex 02`, `03`, and `04` on the corresponding hosts.

### Arguments

| Argument | Meaning |
|---|---|
| `-myindex` | Rank of the local process. Inputs such as `01`, `02`, ... are mapped to zero-based internal ranks. |
| `-list` | Ordered list of participating hostnames. |
| `-iters N` | Number of measured iterations per protocol and message size. |
| `-warmup N` | Number of untimed warm-up iterations. |

At least two hosts are required.

---

## Benchmark Methodology

The benchmark evaluates 11 message sizes:

```text
256 B, 512 B, 1 KiB, 2 KiB, 4 KiB, 8 KiB,
16 KiB, 32 KiB, 64 KiB, 128 KiB, 256 KiB
```

For every message size, both protocols are tested separately.

The work buffer is registered and its remote MR metadata is exchanged **before timing**, so memory registration is not included in the measured collective latency.

Each rank fills its input with:

```text
rank + 1
```

For `P` ranks and a SUM reduction, every output element must therefore equal:

```text
1 + 2 + ... + P = P(P + 1) / 2
```

The benchmark checks this expected value to detect incorrect collective results.

Warm-up iterations are executed before measured iterations, and barriers keep ranks synchronized around benchmark phases.

---

## Performance Metrics

The benchmark reports:

- average All-Reduce latency in **microseconds**
- algorithmic ring bandwidth in **Gbps**
- the faster protocol for each message size

The reported algorithmic traffic per rank is:

```text
2(P - 1) / P * message_size
```

and bandwidth is calculated as:

```text
algorithmic_bandwidth =
    algorithmic_traffic_bits / average_elapsed_time
```

This is an **algorithmic bandwidth metric**, not a direct measurement of physical link utilization.

---

## Experimental Results

The following measurements were collected with:

```text
Iterations per test : 3000
Warm-up per protocol: 200
```

### 2 processes

| Message Size | Eager (us) | Rendezvous (us) | Eager (Gbps) | Rendezvous (Gbps) | Winner |
|---:|---:|---:|---:|---:|:---|
| 256 B | 11.86 | 13.19 | 0.173 | 0.155 | Eager |
| 512 B | 13.49 | 14.35 | 0.304 | 0.285 | Eager |
| 1 KiB | 14.72 | 15.33 | 0.556 | 0.534 | Eager |
| 2 KiB | 12.79 | 14.88 | 1.281 | 1.101 | Eager |
| 4 KiB | 17.56 | 18.39 | 1.866 | 1.782 | Eager |
| 8 KiB | 22.52 | 21.92 | 2.910 | 2.989 | Rendezvous |
| 16 KiB | 28.41 | 25.46 | 4.614 | 5.148 | Rendezvous |
| 32 KiB | 42.05 | 37.41 | 6.234 | 7.008 | Rendezvous |
| 64 KiB | 65.53 | 57.21 | 8.001 | 9.164 | Rendezvous |
| 128 KiB | 108.43 | 102.13 | 9.670 | 10.268 | Rendezvous |
| 256 KiB | 197.12 | 182.04 | 10.639 | 11.520 | Rendezvous |

### 4 processes

| Message Size | Eager (us) | Rendezvous (us) | Eager (Gbps) | Rendezvous (Gbps) | Winner |
|---:|---:|---:|---:|---:|:---|
| 256 B | 25.46 | 26.06 | 0.121 | 0.118 | Eager |
| 512 B | 26.33 | 26.50 | 0.233 | 0.232 | Eager |
| 1 KiB | 26.97 | 27.33 | 0.456 | 0.450 | Eager |
| 2 KiB | 28.24 | 28.82 | 0.870 | 0.853 | Eager |
| 4 KiB | 31.59 | 31.54 | 1.556 | 1.558 | Rendezvous |
| 8 KiB | 38.05 | 36.78 | 2.584 | 2.673 | Rendezvous |
| 16 KiB | 51.17 | 48.34 | 3.842 | 4.068 | Rendezvous |
| 32 KiB | 71.86 | 67.81 | 5.472 | 5.798 | Rendezvous |
| 64 KiB | 108.81 | 99.29 | 7.228 | 7.920 | Rendezvous |
| 128 KiB | 172.84 | 155.75 | 9.100 | 10.099 | Rendezvous |
| 256 KiB | 296.06 | 263.31 | 10.625 | 11.947 | Rendezvous |

### Observed crossover

In both benchmark runs, the program recommends:

```text
16 KiB (16384 bytes)
```

as the **stable protocol crossover**.

The recommendation deliberately does not switch at the first isolated Rendezvous win. Instead, a threshold is accepted only when Rendezvous is at least **5% faster for three consecutive tested message sizes**.

This produces a more conservative crossover than simply selecting the first size where Rendezvous happens to have lower latency.

The measurements show the intended behavior clearly: **Eager performs best for small messages**, while **Rendezvous becomes increasingly advantageous as message size grows**.

---

## Why the Protocols Cross Over

For small messages, fixed synchronization and control costs dominate total latency. Eager already has receive buffers posted and can transfer the message directly through SEND/RECV, which makes it effective in this range.

For larger messages, Rendezvous writes the All-Gather chunk directly into its final remote location. The Eager path receives into a separate registered slot and then performs a `memcpy` into the destination chunk. As payload size grows, avoiding this extra copy becomes increasingly valuable.

The exact crossover is system-dependent; it can change with RNIC, PCIe, CPU, memory, topology, MTU, system load, and benchmark parameters. Therefore, the implementation derives the recommendation from measured results rather than hard-coding a universal threshold.

---

## Important Implementation Details

### Micro-chunk pipelining

`PG_MICRO_CHUNK_SIZE` is set to 4096 elements. During Reduce-Scatter, the next transfer is posted before the CPU reduces the current received micro-chunk.

Conceptually:

```text
receive chunk i
      |
      +--> post transfer of chunk i+1
      |
      +--> reduce chunk i on CPU
```

This creates an opportunity to overlap communication and computation.

### Registered-memory reuse

RDMA memory registration is expensive. The implementation caches the registered work buffer and reuses it when the same buffer and size are used again.

The benchmark also prepares the work buffer outside the timed region.

### Eager receive ring

The Eager protocol maintains a pool of registered receive slots. Completed receive slots are consumed in expected order and immediately re-posted for reuse.

### Rendezvous ordering

The Rendezvous data write and flag write are chained on the same RC send queue. The receiver waits for the epoch flag, while the sender waits only for the signaled flag-write completion.

### Synchronization

A two-phase TCP ring barrier is used for benchmark synchronization:

1. an arrival token circulates around the ring,
2. a release token circulates after all ranks have arrived.

TCP is therefore part of setup/control synchronization, not the measured RDMA payload transfer.

---

## Project Structure

```text
.
├── ring_allreduce.c    # RDMA ring implementation and benchmark
├── Makefile            # Build rules
└── README.md           # Project documentation
```

Optional benchmark screenshots can be stored under a directory such as:

```text
results/
├── results_2processes_a.png
├── results_2processes_b.png
├── results_4processes_a.png
└── results_4processes_b.png
```

---

## Limitations

- Maximum of 4 processes in the current configuration.
- The benchmark currently exercises `PG_INT32` with `PG_SUM`, although the collective implementation also defines additional datatypes and operations.
- The element count must be divisible by the number of processes.
- The Eager All-Gather requires a full ring chunk to fit in one configured Eager receive slot.
- Benchmark results are hardware- and environment-dependent.
- This is an educational/experimental implementation rather than a replacement for production collective libraries such as MPI or NCCL.

---

## Summary

This project implements a complete RDMA Ring All-Reduce and uses it to study a practical protocol-selection problem.

The main design combines:

- ring-based Reduce-Scatter and All-Gather,
- pipelined communication and local reduction,
- two-sided Eager messaging,
- one-sided Rendezvous RDMA writes,
- explicit RDMA resource management,
- reusable registered memory,
- correctness verification,
- and empirical protocol crossover selection.

In the supplied 2-process and 4-process experiments, Eager is preferable for small messages, while Rendezvous becomes consistently faster as messages grow. Using the benchmark's stability criterion, both runs select **16 KiB** as the recommended crossover point.
