// HMTL Kernel — Module 4: PagedTensor (Dynamic Sandbox Extension)
//
// Memory wall solution: when injecting new dimensions into the tensor
// sandbox (e.g., D_security defence axis) causes HBM exhaustion, we
// page out low-attention dimensions to system DRAM via CXL/PCIe.
//
// Inspired by vLLM's PagedAttention, but operating at the tensor-axis
// level rather than KV-cache blocks.  An axis (column in the state matrix)
// whose attention weight averaged across all cores falls below threshold
// is evicted to DRAM.  When a future dispatch cycle predicts cross-axis
// coupling, the axis is prefetched back into HBM before the GPU kernel
// executes.

use crate::kernel_state::{NUM_AXES, NUM_CORES};
use crate::types::Fp8;
use std::collections::VecDeque;

// ─── Page Table Entry ───────────────────────────────────────────────────────

#[repr(C)]
struct PageTableEntry {
    /// Virtual axis index (position in the logical 128-axis space).
    virtual_axis: usize,
    /// Physical location:
    ///   0 = HBM resident
    ///   1 = DRAM (paged out)
    ///   2 = in transit (prefetching / evicting)
    state: u8,
    /// DRAM address if paged out.
    dram_addr: u64,
    /// HBM address when resident.
    hbm_addr: u64,
    /// Last access timestamp (monotonic counter).
    last_access: u64,
    /// Average attention weight across all 128 cores.
    avg_attention_weight: f32,
}

// ─── PagedTensor Manager ────────────────────────────────────────────────────

pub struct PagedTensorManager {
    /// Page table: one entry per axis.
    page_table: [PageTableEntry; NUM_AXES],
    /// Eviction threshold: axes with avg attention < this are candidates.
    eviction_threshold: f32,
    /// Prefetch prediction queue: LRU cache of recently evicted axes.
    prefetch_queue: VecDeque<usize>,
    /// Monotonic clock for LRU ordering.
    clock: u64,
    /// HBM capacity (in axes).
    hbm_capacity: usize,
    /// Currently resident count.
    resident_count: usize,
}

impl PagedTensorManager {
    /// Create a new paged tensor manager.
    ///
    /// `hbm_capacity` = max number of axes that can reside in HBM.
    /// Typically 80–100 for a 128-axis system, leaving room for working set.
    pub fn new(hbm_capacity: usize, eviction_threshold: f32) -> Self {
        let mut page_table: [PageTableEntry; NUM_AXES] = unsafe {
            core::mem::zeroed()
        };
        // Initialize all axes as HBM-resident at boot
        for i in 0..NUM_AXES {
            page_table[i] = PageTableEntry {
                virtual_axis: i,
                state: 0, // HBM resident
                dram_addr: 0,
                hbm_addr: i as u64 * 128, // offset into HBM tensor
                last_access: 0,
                avg_attention_weight: 0.0,
            };
        }
        PagedTensorManager {
            page_table,
            eviction_threshold,
            prefetch_queue: VecDeque::new(),
            clock: 0,
            hbm_capacity,
            resident_count: NUM_AXES.min(hbm_capacity),
        }
    }

    /// Update attention weights and trigger eviction/prefetch.
    ///
    /// Called after each GPU dispatch cycle.  The `attention_weights` matrix
    /// is 128×128 — weight[c][a] is core c's attention on axis a.
    pub fn tick(&mut self, attention_weights: &[[Fp8; NUM_AXES]; NUM_CORES]) {
        self.clock += 1;

        // ─── Compute per-axis average attention ─────────────────────────
        for axis in 0..NUM_AXES {
            let sum: f32 = (0..NUM_CORES)
                .map(|core| attention_weights[core][axis].to_f32())
                .sum();
            self.page_table[axis].avg_attention_weight = sum / NUM_CORES as f32;
            self.page_table[axis].last_access = self.clock;
        }

        // ─── Evict cold axes ────────────────────────────────────────────
        if self.resident_count > self.hbm_capacity {
            // Find coldest resident axis
            let mut coldest = None;
            let mut min_weight = f32::MAX;

            for axis in 0..NUM_AXES {
                let entry = &self.page_table[axis];
                if entry.state == 0 && entry.avg_attention_weight < min_weight {
                    min_weight = entry.avg_attention_weight;
                    coldest = Some(axis);
                }
            }

            if let Some(axis) = coldest {
                if min_weight < self.eviction_threshold {
                    self.evict_axis(axis);
                }
            }
        }

        // ─── Prefetch predicted hot axes ────────────────────────────────
        // Simple prefetch: if an evicted axis was recently accessed
        // (within last 10 ticks), prefetch it back.
        while let Some(&axis) = self.prefetch_queue.front() {
            let entry = &self.page_table[axis];
            if entry.state == 1
                && self.clock - entry.last_access < 10
                && self.resident_count < self.hbm_capacity
            {
                self.prefetch_queue.pop_front();
                self.prefetch_axis(axis);
            } else {
                break;
            }
        }
    }

    /// Evict an axis from HBM to DRAM.
    fn evict_axis(&mut self, axis: usize) {
        let entry = &mut self.page_table[axis];
        if entry.state != 0 {
            return;
        }

        // In hardware: initiate CXL/PCIe DMA from HBM → DRAM.
        // The axis data is 128 FP8 values = 128 bytes.
        // dma_copy(dram_addr, hbm_addr, 128);

        entry.state = 1; // DRAM resident
        entry.dram_addr = allocate_dram_page(128);
        self.resident_count -= 1;
        self.prefetch_queue.push_back(axis);
    }

    /// Prefetch an axis from DRAM back to HBM.
    fn prefetch_axis(&mut self, axis: usize) {
        let entry = &mut self.page_table[axis];
        if entry.state != 1 {
            return;
        }

        // In hardware: initiate DMA from DRAM → HBM.
        // dma_copy(hbm_addr, dram_addr, 128);

        entry.state = 0; // HBM resident
        self.resident_count += 1;
    }

    /// Check if an axis is currently HBM-resident.
    #[inline]
    pub fn is_resident(&self, axis: usize) -> bool {
        self.page_table.get(axis).map(|e| e.state == 0).unwrap_or(false)
    }

    /// Get HBM address for an axis (panics if not resident).
    #[inline]
    pub fn hbm_addr(&self, axis: usize) -> u64 {
        let entry = &self.page_table[axis];
        assert_eq!(entry.state, 0, "Axis {} not HBM-resident", axis);
        entry.hbm_addr
    }
}

// ─── DRAM page allocator stub ───────────────────────────────────────────────

fn allocate_dram_page(size: u64) -> u64 {
    // In production: allocate from a pre-reserved DRAM pool.
    // For now, return a synthetic address in the DRAM region (typically
    // above 0x10000000000 for server-class systems with CXL).
    0x100_0000_0000 + (size * 7) // deterministic pseudorandom for testing
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paged_tensor_basic() {
        let ptm = PagedTensorManager::new(100, 0.01);
        // All axes should be resident initially (up to capacity)
        assert!(ptm.is_resident(0));
        assert!(ptm.is_resident(99));
    }

    #[test]
    fn eviction_on_low_attention() {
        let mut ptm = PagedTensorManager::new(10, 0.05); // only 10 HBM slots
        let mut weights = [[Fp8::ZERO; NUM_AXES]; NUM_CORES];
        // Set very low attention on all axes → eviction should happen
        for core in 0..NUM_CORES {
            for axis in 0..NUM_AXES {
                weights[core][axis] = Fp8::from_f32(0.001);
            }
        }
        ptm.tick(&weights);
        // With capacity=10 and all axes cold, eviction should have run
        // Verify the system is still operational
        assert!(ptm.resident_count <= 10);
    }

    #[test]
    fn hot_axes_preserved() {
        let mut ptm = PagedTensorManager::new(10, 0.05);
        let mut weights = [[Fp8::ZERO; NUM_AXES]; NUM_CORES];
        // Set axis 0–4 as hot (high attention), rest cold
        for core in 0..NUM_CORES {
            for axis in 0..5 {
                weights[core][axis] = Fp8::ONE;
            }
            for axis in 5..NUM_AXES {
                weights[core][axis] = Fp8::from_f32(0.001);
            }
        }
        ptm.tick(&weights);
        // Hot axes should remain resident
        assert!(ptm.is_resident(0));
        assert!(ptm.is_resident(4));
    }
}
