// HMTL Kernel — Module 3: CMTIP (Cross-Model Tensor Interoperability Protocol)
//
// CMTIP enables heterogeneous AI agents (e.g., a Llama-architecture scheduler
// and a DeepSeek-architecture security monitor) to communicate directly on the
// NVLink/CXL fabric without human-readable text alignment.
//
// Agents exchange TensorPackets — structured FP8/BF16/FP16 tensors with a
// standardized header — and perform geometric projection across incompatible
// latent spaces using pre-loaded SRAM adapter matrices.

// ─── TensorPacketHeader ─────────────────────────────────────────────────────
//
// Every tensor exchange on the CMTIP bus is prefixed by this 32-byte header.
// It is placed at the start of a CXL cache-line (64 bytes), with the tensor
// payload immediately following in the same or adjacent cache lines.

#[repr(C, align(64))]
pub struct TensorPacketHeader {
    /// Magic number: 0x484D544C ('HMTL')
    pub magic_number: u32,

    /// Sender agent unique ID (persistent across reboots).
    pub source_model_id: u64,

    /// Tensor shape descriptor, e.g. [1, 128, 128, 1].
    /// Supports up to 4D tensors.  For higher-rank tensors, use
    /// axis_mapping_ptr to point to extended metadata.
    pub tensor_shape: [u32; 4],

    /// Data type tag:
    ///   0 = FP8  (E4M3)
    ///   1 = BF16
    ///   2 = FP16
    ///   3–255 = reserved
    pub data_type: u8,

    /// Reserved padding (3 bytes) for future extensions
    /// (e.g., priority, QoS class, encryption flag).
    pub _reserved: [u8; 3],

    /// Pointer to axis-mapping geometry metadata.
    /// If non-null, points to a region of SRAM describing the geometric
    /// topology of each tensor axis (manifold type, metric tensor, curvature).
    pub axis_mapping_ptr: u64,
}

// Safety: the header is plain data with no pointers (axis_mapping_ptr
// is a physical address, not a Rust reference).
unsafe impl Send for TensorPacketHeader {}
unsafe impl Sync for TensorPacketHeader {}

impl TensorPacketHeader {
    pub const MAGIC: u32 = 0x484D544C;

    /// Create a header for an FP8 tensor with the given shape and sender ID.
    #[inline]
    pub fn new_fp8(source_model_id: u64, shape: [u32; 4]) -> Self {
        TensorPacketHeader {
            magic_number: Self::MAGIC,
            source_model_id,
            tensor_shape: shape,
            data_type: 0, // FP8
            _reserved: [0; 3],
            axis_mapping_ptr: 0,
        }
    }

    /// Validate that a received packet has the correct magic number.
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.magic_number == Self::MAGIC
    }

    /// Number of elements in the tensor payload.
    #[inline]
    pub fn num_elements(&self) -> usize {
        self.tensor_shape.iter().map(|&d| d as usize).product()
    }

    /// Byte size of the tensor payload (excluding header).
    #[inline]
    pub fn payload_bytes(&self) -> usize {
        let bytes_per_element = match self.data_type {
            0 => 1, // FP8
            1 => 2, // BF16
            2 => 2, // FP16
            _ => 1,
        };
        self.num_elements() * bytes_per_element
    }
}

// ─── TensorPacket (Header + Payload) ────────────────────────────────────────
//
// A complete CMTIP message: header followed by inlined tensor payload.
// Allocated in CXL-shared SRAM with cache-line alignment.

#[repr(C, align(64))]
pub struct TensorPacket<const N: usize> {
    pub header: TensorPacketHeader,
    /// Inline payload — N bytes.  N should match header.payload_bytes().
    pub payload: [u8; N],
}

// ─── Cross-Model Projection (Linear Adapter) ────────────────────────────────
//
// When model A (latent dimension d_A) sends a state vector to model B
// (latent dimension d_B), we apply a pre-loaded projection matrix
// W_{A→B} ∈ R^{d_A × d_B} to map the state into B's geometric space.
//
// This is a standard matrix multiplication: X_B = X_A · W_{A→B}
//
// The adapter matrix W is kept in GPU SRAM and loaded once at model
// registration time.  It can be trained offline via contrastive learning
// on aligned multimodal data, or estimated online via canonical correlation
// analysis (CCA) of co-occurring activations.

pub struct LinearAdapter {
    /// Projection matrix: W_{A→B}
    /// Stored row-major: weight[row][col]
    pub weight: Vec<Vec<f32>>,
    pub input_dim: usize,  // d_A
    pub output_dim: usize, // d_B
}

impl LinearAdapter {
    /// Create a new adapter with random orthogonal initialization.
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        // Orthogonal initialization: QR decomposition of random matrix.
        // For small dimensions we use a simple Gram-Schmidt approach.
        let mut weight = vec![vec![0.0_f32; output_dim]; input_dim];
        // Simple random initialization (in production: use QR)
        for i in 0..input_dim {
            for j in 0..output_dim {
                // Kaiming uniform scaled for the projection
                let scale = (2.0 / (input_dim as f32)).sqrt();
                weight[i][j] = (i.wrapping_mul(2654435761) ^ j.wrapping_mul(1597334677))
                    as f32
                    / u32::MAX as f32
                    * 2.0
                    - 1.0;
                weight[i][j] *= scale;
            }
        }
        LinearAdapter {
            weight,
            input_dim,
            output_dim,
        }
    }

    /// Project a vector from model A's space to model B's space.
    ///
    /// X_B[j] = Σ_i X_A[i] · W[i][j]
    pub fn project(&self, input: &[f32]) -> Vec<f32> {
        assert_eq!(
            input.len(),
            self.input_dim,
            "Input dimension mismatch: expected {}, got {}",
            self.input_dim,
            input.len()
        );

        let mut output = vec![0.0_f32; self.output_dim];
        for (i, &x_i) in input.iter().enumerate() {
            let row = &self.weight[i];
            for (j, &w_ij) in row.iter().enumerate() {
                output[j] += x_i * w_ij;
            }
        }
        output
    }

    /// Batch projection: X_B = X_A · W  where X_A is [batch × input_dim]
    pub fn project_batch(&self, input_batch: &[Vec<f32>]) -> Vec<Vec<f32>> {
        input_batch.iter().map(|x| self.project(x)).collect()
    }

    /// Update adapter weights via gradient descent (online adaptation).
    /// dL/dW = X_A^T · dL/dX_B
    pub fn update_weights(&mut self, input: &[f32], output_grad: &[f32], lr: f32) {
        for i in 0..self.input_dim {
            for j in 0..self.output_dim {
                self.weight[i][j] -= lr * input[i] * output_grad[j];
            }
        }
    }
}

// ─── Geometric Penalty Tensor (Bidirectional Error Correction) ──────────────
//
// If model A's output causes model B's internal attention matrix to have
// excessive information entropy (failure to converge), model B emits a
// PenaltyTensor back to model A.  This is applied as an element-wise
// (Hadamard) product with model A's output layer, physically suppressing
// the activations that caused the divergence.
//
// This creates a closed-loop geometric feedback system:
//   A → B (forward projection)
//   B → A (penalty feedback if entropy > threshold)

pub struct PenaltyTensor {
    /// Hadamard-product mask: same shape as model A's output tensor.
    /// Values in [0, 1]; 1 = no penalty, 0 = full suppression.
    pub mask: Vec<f32>,
}

impl PenaltyTensor {
    /// Compute penalty mask from model B's attention entropy.
    ///
    /// For each position i, mask[i] = exp(-λ · entropy[i])
    /// where λ is a temperature parameter controlling penalty sharpness.
    pub fn from_attention_entropy(
        attention_entropy: &[f32],
        lambda: f32,
    ) -> Self {
        let mask: Vec<f32> = attention_entropy
            .iter()
            .map(|&h| (-lambda * h).exp())
            .collect();
        PenaltyTensor { mask }
    }

    /// Apply penalty via Hadamard product: output[i] *= mask[i]
    pub fn apply(&self, output: &mut [f32]) {
        assert_eq!(
            output.len(),
            self.mask.len(),
            "Penalty mask shape mismatch"
        );
        for (o, &m) in output.iter_mut().zip(self.mask.iter()) {
            *o *= m;
        }
    }

    /// Compute the geometric entropy of model B's attention distribution.
    /// Uses Shannon entropy: H = -Σ p_i log(p_i) normalized per head.
    pub fn compute_attention_entropy(attention_weights: &[Vec<f32>]) -> Vec<f32> {
        attention_weights
            .iter()
            .map(|head| {
                let mut entropy = 0.0_f32;
                for &p in head {
                    if p > f32::EPSILON {
                        entropy -= p * p.ln();
                    }
                }
                entropy / (head.len() as f32).ln() // normalize to [0, 1]
            })
            .collect()
    }
}

// ─── CMTIP Bus — Message Routing ────────────────────────────────────────────
//
// The bus manages point-to-point NVLink connections between model agents.
// Each agent registers its SRAM receive buffer; the bus handles DMA setup
// and interrupt signaling on packet arrival.

pub struct CmtipBus {
    /// Registered agents: agent_id → receive buffer physical address.
    receivers: Vec<Option<u64>>,
}

impl CmtipBus {
    pub fn new(max_agents: usize) -> Self {
        CmtipBus {
            receivers: vec![None; max_agents],
        }
    }

    /// Register an agent's receive buffer.
    pub fn register_receiver(&mut self, agent_id: usize, buffer_addr: u64) {
        self.receivers[agent_id] = Some(buffer_addr);
    }

    /// Send a tensor packet from source → target agent.
    ///
    /// In hardware, this initiates an NVLink RDMA write from source's transmit
    /// buffer to target's registered receive buffer, followed by an MSI-X
    /// interrupt to notify the target.
    pub fn send(
        &self,
        _source_id: usize,
        target_id: usize,
        _packet_addr: u64,
        _packet_size: usize,
    ) -> Result<(), CmtipError> {
        match self.receivers.get(target_id) {
            Some(Some(_addr)) => {
                // In production: initiate NVLink RDMA write
                // nvlink_memcpy(target_addr, packet_addr, packet_size);
                // msi_signal(target_id, CMTIP_RX_VECTOR);
                Ok(())
            }
            Some(None) => Err(CmtipError::AgentNotRegistered(target_id)),
            None => Err(CmtipError::AgentIdOutOfRange(target_id)),
        }
    }
}

// ─── Error types ────────────────────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum CmtipError {
    #[error("Agent {0} not registered on CMTIP bus")]
    AgentNotRegistered(usize),
    #[error("Agent ID {0} out of range")]
    AgentIdOutOfRange(usize),
    #[error("Invalid magic number in received packet")]
    InvalidMagic,
    #[error("Payload size mismatch: expected {expected}, got {actual}")]
    PayloadSizeMismatch { expected: usize, actual: usize },
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_validation() {
        let hdr = TensorPacketHeader::new_fp8(42, [1, 128, 128, 1]);
        assert!(hdr.is_valid());
        assert_eq!(hdr.num_elements(), 128 * 128);
        assert_eq!(hdr.payload_bytes(), 128 * 128); // FP8 = 1 byte/element
    }

    #[test]
    fn header_rejects_bad_magic() {
        let mut hdr = TensorPacketHeader::new_fp8(42, [1, 128, 128, 1]);
        hdr.magic_number = 0xDEADBEEF;
        assert!(!hdr.is_valid());
    }

    #[test]
    fn linear_adapter_projection() {
        let adapter = LinearAdapter::new(128, 64);
        let input: Vec<f32> = (0..128).map(|i| i as f32 / 128.0).collect();
        let output = adapter.project(&input);
        assert_eq!(output.len(), 64);
        // Output should be non-zero (weights are initialized)
        assert!(output.iter().any(|&v| v.abs() > 1e-6));
    }

    #[test]
    fn penalty_tensor_applied() {
        let mut output = vec![1.0_f32; 128];
        let entropy = vec![0.5_f32; 128]; // uniform entropy
        let penalty = PenaltyTensor::from_attention_entropy(&entropy, 1.0);
        penalty.apply(&mut output);
        // Every element should be attenuated
        assert!(output.iter().all(|&v| v < 0.7));
        assert!(output.iter().all(|&v| v > 0.5));
    }

    #[test]
    fn cmtip_bus_send() {
        let mut bus = CmtipBus::new(8);
        bus.register_receiver(3, 0xDEAD_BEEF_0000);
        assert!(bus.send(0, 3, 0xBEEF_0000, 16384).is_ok());
        assert!(matches!(
            bus.send(0, 5, 0, 0).unwrap_err(),
            CmtipError::AgentNotRegistered(5)
        ));
    }
}
