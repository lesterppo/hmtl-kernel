// HMTL Kernel — Module 1: Rust Microkernel Shim & CXL State Encoder
//
// This module runs on the CPU host side. It owns system bootstrap,
// hardware interrupt takeover, and physical actuation.
//
// The KernelStateMatrix lives in CXL 3.0 Host-managed Device Memory (HDM),
// shared zero-copy with the GPU. Every interrupt cycle, the CPU writes
// a fresh telemetry snapshot; the GPU reads it, computes an optimal
// dispatch, and writes back an ActuatorCommand.

use crate::types::Fp8;
use core::sync::atomic::{AtomicPtr, Ordering};

// ─── Dimensions ─────────────────────────────────────────────────────────────

/// Number of hardware threads/cores tracked.
pub const NUM_CORES: usize = 128;

/// Number of metric axes per core.
/// Axis layout:
///   0: CPU load (0.0–1.0)
///   1: L2 cache miss rate
///   2: memory lock contention
///   3: I/O wait latency (normalized)
///   4–127: reserved / dynamically extended
pub const NUM_AXES: usize = 128;

// ─── KernelStateMatrix ──────────────────────────────────────────────────────
//
// Stored in a CXL HDM page aligned to 256 bytes (GPU L2 cache-line boundary).
// Every element is FP8 — the native tensor-core data type — so the GPU can
// load the entire 128×128 matrix in a single coalesced transaction.

#[repr(align(256))]
pub struct KernelStateMatrix {
    /// Rows = hardware threads (128). Columns = metric axes (128).
    /// Element FP8: 16,384 bytes total → fits in a single 16 KB GPU L1 block.
    pub data: [[Fp8; NUM_AXES]; NUM_CORES],
}

impl Default for KernelStateMatrix {
    fn default() -> Self {
        KernelStateMatrix {
            data: [[Fp8::ZERO; NUM_AXES]; NUM_CORES],
        }
    }
}

impl KernelStateMatrix {
    /// Total byte size (16,384 for 128×128 FP8).
    pub const SIZE: usize = NUM_CORES * NUM_AXES;

    /// Raw pointer to the matrix data — passed to the GPU as a base address.
    #[inline]
    pub fn as_ptr(&self) -> *const Fp8 {
        self.data.as_ptr() as *const Fp8
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut Fp8 {
        self.data.as_mut_ptr() as *mut Fp8
    }
}

// ─── Telemetry Collector ────────────────────────────────────────────────────
//
// Gathers real-time metrics from the OS and writes them into a KernelStateMatrix.
// In a real deployment this would read MSRs, PMU counters, and scheduler stats.
// The critical invariant: NO conditional branching in the write path — every
// metric slot is always written, even if zero, to keep the memory pipeline
// perfectly predictable.

pub struct TelemetryCollector {
    /// Pointer to the CXL-shared matrix (zero-copy with GPU).
    matrix: AtomicPtr<KernelStateMatrix>,
}

impl TelemetryCollector {
    /// Bind the collector to a CXL-mapped matrix.
    pub fn new(matrix_ptr: *mut KernelStateMatrix) -> Self {
        TelemetryCollector {
            matrix: AtomicPtr::new(matrix_ptr),
        }
    }

    /// Sample current system state and write into the shared matrix.
    ///
    /// # Safety
    /// Caller must ensure the matrix pointer is valid and CXL-mapped.
    /// This function is called from interrupt context (NMI or timer IRQ).
    pub unsafe fn sample(&self) {
        let matrix = self.matrix.load(Ordering::Relaxed);
        let data: &mut [[Fp8; NUM_AXES]; NUM_CORES] = &mut (*matrix).data;

        // ─── Branch-free telemetry write ────────────────────────────────
        // Every core gets every axis written.  The compiler generates a
        // straight-line sequence of stores — no branch predictor pressure.
        for core in 0..NUM_CORES {
            unsafe {
                // Axis 0: CPU load — read from PMU
                data[core][0] = Fp8::from_f32(read_cpu_load(core));
                // Axis 1: L2 cache miss rate
                data[core][1] = Fp8::from_f32(read_l2_miss_rate(core));
                // Axis 2: memory lock contention
                data[core][2] = Fp8::from_f32(read_lock_contention(core));
                // Axis 3: I/O wait latency (normalized μs → [0,1])
                data[core][3] = Fp8::from_f32(read_io_wait(core));
                // Axes 4–127: reserved — written as zero to keep pipeline clean
                for axis in 4..NUM_AXES {
                    data[core][axis] = Fp8::ZERO;
                }
            }
        }
    }
}

// ─── Hardware telemetry stubs ───────────────────────────────────────────────
//
// In production these would read real MSRs / PMU counters via `rdmsr` or
// Linux perf_event_open.  The stubs return realistic synthetic values for
// development and testing.

#[inline(always)]
unsafe fn read_cpu_load(_core: usize) -> f32 {
    // Reads IA32_TIME_STAMP_COUNTER and IA32_MPERF / IA32_APERF MSRs.
    // Stub: uniform load.
    0.5
}

#[inline(always)]
unsafe fn read_l2_miss_rate(_core: usize) -> f32 {
    // Reads L2_MISS / L2_REF PMC events.
    0.03
}

#[inline(always)]
unsafe fn read_lock_contention(_core: usize) -> f32 {
    // Reads kernel lockstat / queue spinlock contention.
    0.01
}

#[inline(always)]
unsafe fn read_io_wait(_core: usize) -> f32 {
    // Reads /proc/stat iowait for the core.
    0.0
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_alignment() {
        let m = KernelStateMatrix::default();
        let addr = &m as *const KernelStateMatrix as usize;
        assert_eq!(addr % 256, 0, "KernelStateMatrix must be 256-byte aligned");
    }

    #[test]
    fn matrix_size() {
        assert_eq!(
            core::mem::size_of::<KernelStateMatrix>(),
            128 * 128,
            "FP8 matrix = exactly 16,384 bytes"
        );
    }

    #[test]
    fn telemetry_collector_no_panic() {
        let mut matrix = KernelStateMatrix::default();
        let collector = TelemetryCollector::new(&mut matrix);
        unsafe { collector.sample(); }
        // Verify non-zero values were written to axis 0
        assert!(matrix.data[0][0].to_f32() > 0.0);
    }
}
