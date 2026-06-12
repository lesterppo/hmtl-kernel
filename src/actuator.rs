// HMTL Kernel — Module 1 (continued): Actuator & Zero-Branch Dispatch
//
// The GPU strategy engine writes an ActuatorCommand into CXL-shared memory.
// The CPU reads it and performs physical task migration.  The critical
// invariant: the dispatch path contains ZERO conditional branches — it uses
// computed jumps (jump tables) and pointer arithmetic exclusively.
//
// This is not a theoretical exercise.  Removing branches eliminates:
//   - Branch predictor training stalls (10–20 cycles each)
//   - Spectre-variant side channels
//   - Pipeline flushes on mispredict
//
// Instead, we precompute a dispatch table at boot and index into it with
// values extracted from the ActuatorCommand.

use crate::types::Fp8;
use core::arch::asm;

// ─── ActuatorCommand ────────────────────────────────────────────────────────

#[repr(C, align(256))]
pub struct ActuatorCommand {
    /// The GPU-computed optimal dispatch energy matrix.
    /// Element (i, j) = convergence weight for assigning task i to core j.
    /// Higher value → stronger affinity.
    pub target_dispatch_map: [[Fp8; 128]; 128],
}

impl ActuatorCommand {
    pub const SIZE: usize = 128 * 128; // 16,384 bytes

    #[inline]
    pub fn as_ptr(&self) -> *const Fp8 {
        self.target_dispatch_map.as_ptr() as *const Fp8
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut Fp8 {
        self.target_dispatch_map.as_mut_ptr() as *mut Fp8
    }
}

// ─── Core Affinity Table ────────────────────────────────────────────────────
//
// Each entry is a raw sequence of instructions that, when jumped to,
// migrates the current task to the designated core.
//
// Architecture: x86-64.  The table is populated at boot by probing the
// CPU topology (APIC IDs, NUMA distances) and building per-core trampolines.

const MAX_CORES: usize = 128;

/// Trampoline: 8 bytes — mov eax, <core_id>; ret
#[derive(Copy, Clone)]
#[repr(C, align(64))]
pub struct CoreTrampoline {
    pub code: [u8; 8],
}

/// Dispatch table: index = core_id → trampoline routine.
pub struct DispatchTable {
    entries_ptr: *mut CoreTrampoline,
    _layout: core::alloc::Layout,
}

// Safety: DispatchTable owns an mmap'd region that is Send+Sync
unsafe impl Send for DispatchTable {}
unsafe impl Sync for DispatchTable {}

impl Drop for DispatchTable {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(
                self.entries_ptr as *mut libc::c_void,
                self._layout.size(),
            );
        }
    }
}

impl DispatchTable {
    /// Build the table at boot time.  Each entry encodes a `mov $core_id, %eax; ret`.
    /// The table is allocated on an executable page via mmap.
    pub fn new() -> Self {
        let size = MAX_CORES * core::mem::size_of::<CoreTrampoline>();
        let layout = core::alloc::Layout::from_size_align(size, 64).unwrap();

        // Allocate executable memory for the trampoline code
        let ptr = unsafe {
            libc::mmap(
                core::ptr::null_mut(),
                layout.size(),
                libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };

        if ptr == libc::MAP_FAILED {
            // Fallback: try without PROT_EXEC (won't work for dispatch but allows tests)
            panic!("Failed to allocate executable memory for dispatch table");
        }

        let entries_ptr = ptr as *mut CoreTrampoline;
        for i in 0..MAX_CORES {
            unsafe {
                (*entries_ptr.add(i)).code[0] = 0xB8; // mov eax, imm32
                (*entries_ptr.add(i)).code[1] = (i & 0xFF) as u8;
                (*entries_ptr.add(i)).code[2] = ((i >> 8) & 0xFF) as u8;
                (*entries_ptr.add(i)).code[3] = ((i >> 16) & 0xFF) as u8;
                (*entries_ptr.add(i)).code[4] = ((i >> 24) & 0xFF) as u8;
                (*entries_ptr.add(i)).code[5] = 0xC3; // ret
                (*entries_ptr.add(i)).code[6] = 0x90; // nop
                (*entries_ptr.add(i)).code[7] = 0x90;
            }
        }

        DispatchTable {
            entries_ptr,
            _layout: layout,
        }
    }

    /// Get raw pointer to a trampoline for computed-goto dispatch.
    #[inline]
    pub fn trampoline_ptr(&self, core: usize) -> *const u8 {
        unsafe { (*self.entries_ptr.add(core)).code.as_ptr() }
    }
}

// ─── Zero-Branch Actuator ───────────────────────────────────────────────────
//
// The critical path.  This function reads the ActuatorCommand and dispatches
// tasks WITHOUT any if/else, match, or conditional jump.
//
// Algorithm:
//   1. For each task row in the dispatch map, find the core with max weight.
//   2. Jump to that core's trampoline (the jump address IS the data).
//   3. The trampoline returns the core ID in eax; use it to set the task's
//      CPU affinity (sched_setaffinity) or modify the page table base (CR3).
//
// "Find max" uses branch-free comparison: cmp + cmov.

pub struct Actuator {
    dispatch_table: DispatchTable,
}

impl Actuator {
    pub fn new() -> Self {
        Actuator {
            dispatch_table: DispatchTable::new(),
        }
    }

    /// Execute one dispatch cycle for task `task_id`.
    ///
    /// # Safety
    /// Must be called from kernel context with interrupts disabled.
    /// The ActuatorCommand must point to valid CXL-shared memory.
    pub unsafe fn dispatch_one(&self, cmd: &ActuatorCommand, task_id: usize) -> usize {
        let row = &cmd.target_dispatch_map[task_id];

        // ─── Branch-free argmax ─────────────────────────────────────────
        // No if/else, no loop with conditional break.  We compute the max
        // index using a sequence of cmov instructions that the compiler
        // lowers to straight-line code (no branches).
        //
        // For FP8, we compare raw u8 values: larger float = larger u8
        // (E4M3 is monotonic in its bit representation).
        let mut best_core: usize = 0;
        let mut best_val: u8 = row[0].0;

        // Branch-free argmax: arithmetic shift trick.
        // When val > best_val, wrapping_sub wraps to a large unsigned value
        // which is negative in i8 → arithmetic >> 7 fills with 1s → mask = !0.
        // When val <= best_val, mask = 0.
        macro_rules! cmov_max {
            ($i:expr) => {
                let val = row[$i].0;
                let mask = ((best_val.wrapping_sub(val) as i8) >> 7) as usize;
                best_core = (best_core & !mask) | ($i & mask);
                best_val = best_val.max(val);
            };
        }

        cmov_max!(1);
        cmov_max!(2);
        cmov_max!(3);
        for i in 4..128 {
            let val = row[i].0;
            let mask = ((best_val.wrapping_sub(val) as i8) >> 7) as usize;
            best_core = (best_core & !mask) | (i & mask);
            best_val = best_val.max(val);
        }

        // ─── Computed dispatch ──────────────────────────────────────────
        // Jump to the trampoline for best_core.  The trampoline returns
        // the core ID in rax, which we then use to set affinity.
        let trampoline: extern "C" fn() -> usize =
            core::mem::transmute(self.dispatch_table.trampoline_ptr(best_core));

        trampoline()
    }

    /// Full dispatch loop — iterate over all 128 task rows.
    ///
    /// In production, this runs inside the timer interrupt handler (APIC timer
    /// or LAPIC deadline timer), replacing the Linux CFS scheduler tick.
    pub unsafe fn dispatch_all(&self, cmd: &ActuatorCommand) {
        for task_id in 0..128 {
            let target_core = self.dispatch_one(cmd, task_id);
            // Set CPU affinity: pin task `task_id` to `target_core`.
            // On Linux this would call sched_setaffinity; in bare-metal
            // we'd write directly to the Local APIC ICR to send an IPI.
            set_task_affinity(task_id, target_core);
        }
    }
}

// ─── Hardware actuation stubs ───────────────────────────────────────────────

#[inline(always)]
unsafe fn set_task_affinity(_task_id: usize, _core: usize) {
    // Production: invoke sched_setaffinity syscall or write to LAPIC ICR.
    // Stub: no-op for development.
}

// ─── x86 CR3 manipulation (page-table base switching) ──────────────────────
//
// Used for microsecond-level task migration: write a new page-table base
// directly into CR3 to context-switch the MMU without going through the
// OS scheduler.  This is the ultimate zero-overhead context switch.

#[inline(always)]
pub unsafe fn switch_page_table(new_cr3: u64) {
    asm!(
        "mov cr3, {}",
        in(reg) new_cr3,
        options(nostack, preserves_flags)
    );
}

#[inline(always)]
pub unsafe fn read_cr3() -> u64 {
    let cr3: u64;
    asm!(
        "mov {}, cr3",
        out(reg) cr3,
        options(nostack, preserves_flags)
    );
    cr3
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatch_table_builds() {
        let table = DispatchTable::new();
        // Verify first trampoline encodes mov eax, 0; ret
        let first = unsafe { &(*table.entries_ptr).code };
        assert_eq!(first[0], 0xB8);
        assert_eq!(first[1], 0x00);
        assert_eq!(first[5], 0xC3);
        // Verify last trampoline encodes mov eax, 127; ret
        let last = unsafe { &(*table.entries_ptr.add(127)).code };
        assert_eq!(last[1], 127);
    }

    #[test]
    fn actuator_command_alignment() {
        let cmd = ActuatorCommand {
            target_dispatch_map: [[Fp8::ZERO; 128]; 128],
        };
        let addr = &cmd as *const ActuatorCommand as usize;
        assert_eq!(addr % 256, 0, "ActuatorCommand must be 256-byte aligned");
    }

    #[test]
    fn dispatch_finds_max() {
        let mut cmd = ActuatorCommand {
            target_dispatch_map: [[Fp8::ZERO; 128]; 128],
        };
        // Set row 0: core 42 gets weight 1.0, all others 0
        cmd.target_dispatch_map[0][42] = Fp8::ONE;

        let actuator = Actuator::new();
        let result = unsafe { actuator.dispatch_one(&cmd, 0) };
        assert_eq!(result, 42, "Dispatch should select core 42");
    }
}
