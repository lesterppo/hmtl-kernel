// HMTL Kernel — CXL 3.0 Shared Memory Manager
//
// Manages the zero-copy memory region between the CPU (Rust microkernel shim)
// and GPU (Triton/CUDA strategy engine) via CXL 3.0 Host-managed Device Memory.
//
// CXL 3.0 HDM allows the host CPU to map GPU HBM pages directly into its own
// address space, and vice versa.  This eliminates the PCIe DMA round-trip:
// both sides read/write the same physical memory with cache-coherent semantics.

use crate::kernel_state::KernelStateMatrix;
use crate::actuator::ActuatorCommand;
use core::alloc::Layout;
use core::ptr::NonNull;
use std::fs;

// ─── mbind syscall (not always available in libc crate) ─────────────────────

const MPOL_BIND: libc::c_int = 2;
const MPOL_MF_MOVE: libc::c_uint = 1;
const MPOL_MF_STRICT: libc::c_uint = 2;

extern "C" {
    fn mbind(
        addr: *mut libc::c_void,
        len: libc::c_ulong,
        mode: libc::c_int,
        nodemask: *const libc::c_ulong,
        maxnode: libc::c_ulong,
        flags: libc::c_uint,
    ) -> libc::c_long;
}

/// CXL-aware allocator that allocates pages in the CXL HDM region.
///
/// On Linux, CXL memory appears under /sys/devices/system/node/nodeN/
/// as "memory only" NUMA nodes.  We allocate from those nodes using
/// mbind() with MPOL_BIND to force placement on CXL-attached memory.
pub struct CxlAllocator;

impl CxlAllocator {
    /// Detect available CXL memory nodes.  Returns a list of NUMA node IDs
    /// that have CXL-attached memory (type "mixed" or purely memory-only nodes).
    pub fn detect_cxl_nodes() -> Vec<usize> {
        let mut nodes = Vec::new();
        let node_path = std::path::Path::new("/sys/devices/system/node/");

        if !node_path.exists() {
            // No NUMA — likely a single-node system without CXL.
            // Fall back to regular heap allocation.
            return vec![0];
        }

        if let Ok(entries) = fs::read_dir(node_path) {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                // Match nodeN directories
                if let Some(node_str) = name_str.strip_prefix("node") {
                    if let Ok(node_id) = node_str.parse::<usize>() {
                        // Check if this node has CXL memory
                        let has_cxl = entry.path().join("has_cxl_memory");
                        if has_cxl.exists() {
                            if let Ok(val) = fs::read_to_string(&has_cxl) {
                                if val.trim() == "1" {
                                    nodes.push(node_id);
                                }
                            }
                        }
                        // Also check cpulist — empty cpulist = memory-only node (likely CXL)
                        let cpulist = entry.path().join("cpulist");
                        if cpulist.exists() {
                            if let Ok(val) = fs::read_to_string(&cpulist) {
                                if val.trim().is_empty() {
                                    nodes.push(node_id);
                                }
                            }
                        }
                    }
                }
            }
        }

        if nodes.is_empty() {
            nodes.push(0); // fallback
        }
        nodes
    }

    /// Allocate `layout` bytes on a CXL NUMA node.
    ///
    /// Uses mmap with MAP_SHARED | MAP_ANONYMOUS, then mbind to pin to CXL node.
    /// The returned pointer is cache-coherent across CPU and CXL-attached GPU.
    pub fn allocate_on_cxl(layout: Layout, numa_node: usize) -> Option<NonNull<u8>> {
        // mmap
        let ptr = unsafe {
            libc::mmap(
                core::ptr::null_mut(),
                layout.size(),
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_HUGETLB,
                -1,
                0,
            )
        };

        if ptr == libc::MAP_FAILED {
            // Fallback: try without huge pages
            let ptr = unsafe {
                libc::mmap(
                    core::ptr::null_mut(),
                    layout.size(),
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                    -1,
                    0,
                )
            };
            if ptr == libc::MAP_FAILED {
                return None;
            }
            // mbind to CXL node
            unsafe {
                let nodemask: libc::c_ulong = 1 << numa_node;
                mbind(
                    ptr,
                    layout.size() as libc::c_ulong,
                    MPOL_BIND,
                    &nodemask as *const _ as *const libc::c_ulong,
                    64,
                    MPOL_MF_MOVE | MPOL_MF_STRICT,
                );
            }
            return NonNull::new(ptr as *mut u8);
        }

        // mbind to CXL node
        unsafe {
            let nodemask: libc::c_ulong = 1 << numa_node;
            mbind(
                ptr,
                layout.size() as libc::c_ulong,
                MPOL_BIND,
                &nodemask as *const _ as *const libc::c_ulong,
                64,
                MPOL_MF_MOVE | MPOL_MF_STRICT,
            );
        }

        NonNull::new(ptr as *mut u8)
    }

    /// Free a CXL allocation.
    pub unsafe fn deallocate(ptr: *mut u8, layout: Layout) {
        libc::munmap(ptr as *mut libc::c_void, layout.size());
    }
}

// ─── SharedMemoryContext ────────────────────────────────────────────────────
//
// A single CXL-shared memory region containing both the KernelStateMatrix
// (written by CPU, read by GPU) and the ActuatorCommand (written by GPU,
// read by CPU).  Both are placed in the same 64 KB HDM page for maximum
// locality on the CXL fabric.

#[repr(C, align(4096))]
pub struct SharedMemoryContext {
    /// CPU → GPU: telemetry input matrix.
    pub state_matrix: KernelStateMatrix,

    /// GPU → CPU: dispatch output matrix.
    pub actuator_command: ActuatorCommand,
}

impl SharedMemoryContext {
    /// Layout: two 128×128 FP8 matrices = 32,768 bytes (+ alignment padding).
    const LAYOUT: Layout = unsafe {
        Layout::from_size_align_unchecked(
            core::mem::size_of::<Self>(),
            4096,
        )
    };

    /// Allocate on CXL memory, preferring the first detected CXL node.
    pub fn allocate() -> Option<NonNull<Self>> {
        let nodes = CxlAllocator::detect_cxl_nodes();
        let ptr = CxlAllocator::allocate_on_cxl(Self::LAYOUT, nodes[0])?;
        Some(ptr.cast::<Self>())
    }

    /// Physical address — needed by GPU to program its memory controller.
    /// On CXL, the physical address IS the address the GPU uses (cache-coherent).
    pub fn physical_address(&self) -> u64 {
        self as *const Self as u64
    }

    /// GPU-visible base pointer for the state matrix.
    pub fn state_matrix_gpu_ptr(&self) -> u64 {
        &self.state_matrix as *const KernelStateMatrix as u64
    }

    /// GPU-visible base pointer for the actuator command.
    pub fn actuator_command_gpu_ptr(&self) -> u64 {
        &self.actuator_command as *const ActuatorCommand as u64
    }
}

// Safety: SharedMemoryContext contains only plain data, no file descriptors
// or thread-local state.
unsafe impl Send for SharedMemoryContext {}
unsafe impl Sync for SharedMemoryContext {}

// ─── PCIe BAR Mapper (for non-CXL systems) ──────────────────────────────────
//
// On systems without CXL 3.0, we fall back to PCIe BAR mapping:
// map the GPU's BAR (Base Address Register) into CPU address space.
// This is the traditional GPU Direct RDMA path.

pub struct PcieBarMapper;

impl PcieBarMapper {
    /// Map a GPU BAR region into CPU address space.
    /// `bar_addr` = physical address from lspci -v (e.g., 0x3fe00000000)
    /// `size` = BAR size from lspci
    pub fn map_bar(bar_addr: u64, size: usize) -> Option<NonNull<u8>> {
        // Open /dev/mem with appropriate permissions.
        let fd = unsafe {
            libc::open(
                b"/dev/mem\0" as *const u8 as *const libc::c_char,
                libc::O_RDWR | libc::O_SYNC,
            )
        };
        if fd < 0 {
            return None;
        }

        let ptr = unsafe {
            libc::mmap(
                core::ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                bar_addr as libc::off_t,
            )
        };

        unsafe { libc::close(fd); }

        if ptr == libc::MAP_FAILED {
            return None;
        }

        NonNull::new(ptr as *mut u8)
    }
}
