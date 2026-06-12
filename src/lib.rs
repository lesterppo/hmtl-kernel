// HMTL (Hardware-Mapped Tensor Language) Autonomous OS Kernel
//
// A silicon-native operating system kernel that replaces human-readable
// control flow with matrix-multiplication-based geometric convergence.
//
// Architecture:
//   Module 1: Rust microkernel shim (CPU side) — telemetry + actuation
//   Module 2: GPU HMTL strategy engine (Triton/CUDA) — tensor-core dispatch
//   Module 3: CMTIP cross-model protocol — heterogeneous agent communication
//   Module 4: PagedTensor — dynamic axis-level memory swapping
//
// All modules communicate via CXL 3.0 zero-copy shared memory.

pub mod types;
pub mod kernel_state;
pub mod memory;
pub mod actuator;
pub mod cmtip;
pub mod paged_tensor;
pub mod telemetry_real;

// Re-export primary public types.
pub use types::Fp8;
pub use kernel_state::{KernelStateMatrix, TelemetryCollector, NUM_AXES, NUM_CORES};
pub use memory::{SharedMemoryContext, CxlAllocator};
pub use actuator::{Actuator, ActuatorCommand, DispatchTable, CoreTrampoline};
pub use cmtip::{TensorPacketHeader, TensorPacket, LinearAdapter, PenaltyTensor, CmtipBus, CmtipError};
pub use paged_tensor::PagedTensorManager;
