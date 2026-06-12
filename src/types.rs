// HMTL Kernel — Core Type Definitions
//
// FP8 (8-bit floating point) type for hardware-aligned tensor storage.
// FP8 E4M3 format: 1 sign, 4 exponent, 3 mantissa bits.
// Used throughout the kernel for cache-line efficiency on CXL 3.0 fabric.

use core::fmt;
use core::ops;

/// FP8 (E4M3) — 8-bit floating point backed by u8.
///
/// Layout: S[7] | EEEEE[6:3] | MMM[2:0]
///
/// Triton equivalent: `tl.float8e4b8`
/// CUDA equivalent: `__nv_fp8_e4m3`
#[derive(Clone, Copy, PartialEq, Eq, Hash, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(transparent)]
pub struct Fp8(pub u8);

// ─── FP8 Constants ──────────────────────────────────────────────────────────

impl Fp8 {
    pub const ZERO: Self = Fp8(0x00);
    pub const ONE: Self = Fp8(0x38);   // 1.0 in E4M3: exp=0111 (7), bias=7, mant=000
    pub const NEG_ONE: Self = Fp8(0xB8); // -1.0
    pub const MIN: Self = Fp8(0xFE);   // -448.0
    pub const MAX: Self = Fp8(0x7E);   // 448.0
    pub const NAN: Self = Fp8(0x7F);
    pub const INFINITY: Self = Fp8(0x78);
    pub const NEG_INFINITY: Self = Fp8(0xF8);

    /// Convert FP8 → f32 for CPU-side operations.
    #[inline]
    pub fn to_f32(self) -> f32 {
        let bits = self.0;
        if bits == 0x00 {
            return 0.0;
        }
        if bits == 0x80 {
            return -0.0;
        }
        // NaN/Inf detection
        let exp = (bits >> 3) & 0x0F;
        let mant = bits & 0x07;
        if exp == 0x0F {
            if mant == 0 {
                return if (bits & 0x80) != 0 { f32::NEG_INFINITY } else { f32::INFINITY };
            }
            return f32::NAN;
        }
        // Normal: bias=7
        let sign = if (bits & 0x80) != 0 { -1.0_f32 } else { 1.0_f32 };
        if exp == 0 {
            // Subnormal
            sign * (mant as f32) * 2.0_f32.powi(-6)
        } else {
            sign * (1.0 + (mant as f32) / 8.0) * 2.0_f32.powi((exp as i32) - 7)
        }
    }

    /// Convert f32 → FP8 (E4M3), saturating/clamping.
    #[inline]
    pub fn from_f32(x: f32) -> Self {
        if x.is_nan() {
            return Self::NAN;
        }
        if x >= 448.0 {
            return Self::MAX;
        }
        if x <= -448.0 {
            return Self::MIN;
        }
        let bits = x.to_bits();
        let sign = (bits >> 31) as u8;
        let mut exp = (((bits >> 23) & 0xFF) as i32) - 127 + 7; // adjust bias
        let mut mant = (bits & 0x007F_FFFF) >> 20; // top 3 bits of 23-bit mantissa
        if exp < -6 {
            // Subnormal
            let shift = -6 - exp;
            if shift > 3 {
                return if sign != 0 { Self::NEG_ONE } else { Self::ZERO };
            }
            mant = (0x04 >> shift) | (mant >> (shift + 1));
            exp = 0;
        } else if exp > 14 {
            return if sign != 0 { Self::MIN } else { Self::MAX };
        }
        let encoded = (sign << 7) | (((exp as u8) & 0x0F) << 3) | ((mant as u8) & 0x07);
        Fp8(encoded)
    }
}

// ─── Display / Debug ────────────────────────────────────────────────────────

impl fmt::Display for Fp8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_f32())
    }
}

impl fmt::Debug for Fp8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Fp8({:.3})", self.to_f32())
    }
}

// ─── Arithmetic on the CPU (for testing/routing only — not the hot path) ────

impl ops::Add for Fp8 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self { Fp8::from_f32(self.to_f32() + rhs.to_f32()) }
}

impl ops::Mul for Fp8 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self { Fp8::from_f32(self.to_f32() * rhs.to_f32()) }
}

// Convenience: build a 128×128 array of Fp8 zero.
#[inline]
pub fn zero_state_matrix() -> [[Fp8; 128]; 128] {
    [[Fp8::ZERO; 128]; 128]
}
