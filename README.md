# HMTL — Hardware-Mapped Tensor Language Kernel

**Silicon-native OS kernel for tensor-core dispatch with real Linux telemetry.**

[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange)](https://rust-lang.org)
[![Tests](https://img.shields.io/badge/tests-21%20passed-brightgreen)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

## What It Is

A Rust microkernel that replaces human-readable control flow with matrix-multiplication-based geometric convergence. The GPU strategy engine reads real system telemetry, computes optimal task dispatch via tensor-core matrix multiplication, and writes actuator commands — all with zero conditional branches on the dispatch path.

```
CPU Telemetry ──→ CXL Shared Memory ──→ GPU Strategy Engine
                                            │
                                      Tensor Core GEMM
                                      (128×128 × 128×128)
                                            │
                                            ▼
CPU Scheduler ◀── CXL Shared Memory ◀── Actuator Command
        │
        ▼
   Zero-Branch Dispatch (computed jumps)
```

## Architecture

```
Module 1: Rust Microkernel Shim (CPU)
  ├── kernel_state.rs    KernelStateMatrix (128×128 FP8), telemetry
  ├── actuator.rs        Zero-branch dispatch (trampoline tables, CR3 switch)
  └── telemetry_real.rs  Real hardware data (/proc/stat, /proc/meminfo)

Module 2: GPU Strategy Engine (Triton/CUDA)
  └── kernels/hmtl_kernel.py   Tensor-core GEMM, FP8 conversion, dim folding

Module 3: CMTIP (Cross-Model Protocol)
  └── cmtip.rs      TensorPacket, LinearAdapter, PenaltyTensor, CmtipBus

Module 4: PagedTensor (Memory Manager)
  └── paged_tensor.rs    Axis-level eviction, HBM↔DRAM paging, attention-weighted LRU

Shared Memory: CXL 3.0 HDM
  └── memory.rs     CxlAllocator (mbind), SharedMemoryContext, PCIe BAR mapper
```

## Quick Start

```bash
git clone https://github.com/lesterppo/hmtl-kernel
cd hmtl-kernel
cargo test    # 21 tests
```

```bash
# Python GPU kernel (requires Triton + CUDA)
python3 kernels/hmtl_kernel.py
# → Module 2: All 7 validations passed
```

## Real Hardware Telemetry

The kernel reads actual Linux system data — no stubs:

| Metric | Source | What It Reads |
|--------|--------|---------------|
| CPU load (per core) | `/proc/stat` | Delta of non-idle/total jiffies |
| Memory pressure | `/proc/meminfo` | `MemAvailable / MemTotal` |
| Swap pressure | `/proc/meminfo` | `SwapFree / SwapTotal` |
| IO wait | `/proc/diskstats` | IO ticks / total ticks across disks |
| Load average | `/proc/loadavg` | 1-min load / num_cpus |

```rust
use hmtl_kernel::telemetry_real::*;

let mut cpu = CpuLoadReader::new(num_cpus::get());
let memory_pressure = MemoryReader::read_pressure();
let io_util = IoReader::read_io_utilization();

// Populate the CXL-shared state matrix with real data
let mut matrix = KernelStateMatrix::default();
populate_real_telemetry(&mut matrix, &mut cpu);
```

## FP8 (E4M3) Type

8-bit floating point for tensor-core efficiency. 128×128 matrix = 16,384 bytes — fits in a single GPU L1 cache block.

```rust
use hmtl_kernel::types::Fp8;

let a = Fp8::from_f32(3.14);
let b = Fp8::from_f32(2.0);
let c = a * b;        // ~6.25
let f: f32 = c.to_f32();

// Constants: ZERO, ONE, NEG_ONE, MIN(-448), MAX(448), NAN, INFINITY
```

**Precision:** 4.3% mean relative error (expected 3-6% for 3-bit mantissa).

## Zero-Branch Dispatch

The CPU dispatch path contains **zero conditional branches**. Instead of `if/else` or priority queues, it uses:

1. **Branch-free argmax** — arithmetic shift trick (`wrapping_sub` as `i8 >> 7`) for `cmov`-style comparison
2. **Computed jumps** — pre-built trampoline table (one `mov $core_id, %eax; ret` per core)
3. **CR3 manipulation** — direct page-table base switching for microsecond context migration

```rust
let actuator = Actuator::new();
let cmd = ActuatorCommand { /* GPU-computed dispatch map */ };
unsafe {
    actuator.dispatch_one(&cmd, task_id);  // Zero branches
}
```

## Module Overview

| Module | File | Purpose |
|--------|------|---------|
| Types | `src/types.rs` | FP8 E4M3 type with full arithmetic, constants |
| Kernel State | `src/kernel_state.rs` | 128×128 FP8 matrix, telemetry collector |
| Telemetry | `src/telemetry_real.rs` | Real Linux hardware data (7 tests) |
| Memory | `src/memory.rs` | CXL 3.0 allocation, shared memory context |
| Actuator | `src/actuator.rs` | Zero-branch dispatch, trampoline tables, CR3 |
| CMTIP | `src/cmtip.rs` | Tensor packet, linear adapter, penalty feedback, bus |
| PagedTensor | `src/paged_tensor.rs` | Axis-level eviction/prefetch, LRU, HBM↔DRAM |
| GPU Kernel | `kernels/hmtl_kernel.py` | Triton tensor-core GEMM, FP8 conversion |

## Tests

```bash
cargo test
# → 21 passed (2 suites)

# Includes:
#   - FP8 conversion & arithmetic
#   - Matrix alignment (256-byte)
#   - Linear adapter projection
#   - Penalty tensor application
#   - CMTIP bus send/receive
#   - Paged tensor eviction & hot axis preservation
#   - Dispatch table construction
#   - Actuator argmax selection
#   - Real telemetry (CPU, memory, swap, IO, loadavg)
```

## Related

- **[cmtip](https://github.com/lesterppo/cmtip)** — Cross-Model Tensor Interoperability Protocol: Python library with gRPC bus, 16 LLM tools, semantic memory, self-improving adapters

## License

MIT
