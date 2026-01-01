//! Walsh-Hadamard rotation for embedding preprocessing.
//!
//! Applies a fast, structured rotation: x' = H(D ⊙ x)
//! where D is random ±1 diagonal and H is a Hadamard-like transform.

use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use rand::Rng;

/// Rotator for embedding preprocessing
pub struct Rotator {
    /// Original dimension
    dim: u32,
    /// Padded dimension (power of 2)
    padded_dim: u32,
    /// Random signs for diagonal D
    signs: Vec<f32>,
}

impl Rotator {
    /// Seed for deterministic rotations
    const ROTATE_SEED: u64 = 0xDEAD_BEEF_CAFE_1234;
    
    /// Create a new rotator for the given dimension
    pub fn new(dim: u32) -> Self {
        let padded_dim = Self::next_power_of_two(dim);
        
        // Generate random signs
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(Self::ROTATE_SEED);
        let signs: Vec<f32> = (0..padded_dim)
            .map(|_| if rng.r#gen::<bool>() { 1.0 } else { -1.0 })
            .collect();
        
        Self {
            dim,
            padded_dim,
            signs,
        }
    }

    /// Find next power of two >= n
    fn next_power_of_two(n: u32) -> u32 {
        let mut p = 1u32;
        while p < n {
            p *= 2;
        }
        p
    }

    /// Apply rotation: x' = H(D ⊙ x)
    pub fn rotate(&self, x: &[f32]) -> Vec<f32> {
        assert!(x.len() <= self.padded_dim as usize);
        
        // Pad with zeros if needed
        let mut v = vec![0.0f32; self.padded_dim as usize];
        for (i, &val) in x.iter().enumerate() {
            v[i] = val * self.signs[i];
        }
        
        // Apply Walsh-Hadamard transform in-place
        self.hadamard_transform(&mut v);
        
        // Return only the original dimensions
        v.truncate(self.dim as usize);
        
        // Normalize
        let norm_factor = 1.0 / (self.padded_dim as f32).sqrt();
        for val in &mut v {
            *val *= norm_factor;
        }
        
        v
    }

    /// Apply inverse rotation
    pub fn rotate_inverse(&self, x: &[f32]) -> Vec<f32> {
        // Hadamard is its own inverse (up to scaling)
        // For inverse: x = D ⊙ H(x') * scale
        
        let mut v = vec![0.0f32; self.padded_dim as usize];
        for (i, &val) in x.iter().enumerate() {
            v[i] = val;
        }
        
        // Apply Hadamard
        self.hadamard_transform(&mut v);
        
        // Apply inverse signs
        for i in 0..self.dim as usize {
            v[i] *= self.signs[i];
        }
        
        // Normalize
        let norm_factor = 1.0 / (self.padded_dim as f32).sqrt();
        for val in &mut v {
            *val *= norm_factor;
        }
        
        v.truncate(self.dim as usize);
        v
    }

    /// Fast Walsh-Hadamard transform in-place
    fn hadamard_transform(&self, v: &mut [f32]) {
        let n = v.len();
        assert!(n.is_power_of_two());
        
        let mut h = 1;
        while h < n {
            for i in (0..n).step_by(h * 2) {
                for j in i..(i + h) {
                    let x = v[j];
                    let y = v[j + h];
                    v[j] = x + y;
                    v[j + h] = x - y;
                }
            }
            h *= 2;
        }
    }

    /// Get original dimension
    pub fn dim(&self) -> u32 {
        self.dim
    }

    /// Get padded dimension
    pub fn padded_dim(&self) -> u32 {
        self.padded_dim
    }
}

/// Block-Hadamard for non-power-of-two dimensions
/// Applies Hadamard to blocks of size 2^k
pub struct BlockRotator {
    dim: u32,
    block_size: u32,
    num_blocks: u32,
    signs: Vec<f32>,
}

impl BlockRotator {
    /// Seed for deterministic block rotations
    const BLOCK_ROT_SEED: u64 = 0xB10C_B0A7_CAFE_5678;
    
    /// Create a block rotator with specified block size
    pub fn new(dim: u32, block_size: u32) -> Self {
        assert!(block_size.is_power_of_two());
        let num_blocks = (dim + block_size - 1) / block_size;
        
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(Self::BLOCK_ROT_SEED);
        let total_size = num_blocks * block_size;
        let signs: Vec<f32> = (0..total_size)
            .map(|_| if rng.r#gen::<bool>() { 1.0 } else { -1.0 })
            .collect();
        
        Self {
            dim,
            block_size,
            num_blocks,
            signs,
        }
    }

    /// Apply block rotation
    pub fn rotate(&self, x: &[f32]) -> Vec<f32> {
        let mut result = vec![0.0f32; self.dim as usize];
        
        for block_idx in 0..self.num_blocks as usize {
            let start = block_idx * self.block_size as usize;
            let end = (start + self.block_size as usize).min(self.dim as usize);
            
            // Pad block
            let mut block = vec![0.0f32; self.block_size as usize];
            for (i, idx) in (start..end).enumerate() {
                if idx < x.len() {
                    block[i] = x[idx] * self.signs[start + i];
                }
            }
            
            // Apply Hadamard to block
            Self::hadamard_transform_block(&mut block);
            
            // Copy back
            let norm = 1.0 / (self.block_size as f32).sqrt();
            for (i, idx) in (start..end).enumerate() {
                result[idx] = block[i] * norm;
            }
        }
        
        result
    }

    fn hadamard_transform_block(v: &mut [f32]) {
        let n = v.len();
        let mut h = 1;
        while h < n {
            for i in (0..n).step_by(h * 2) {
                for j in i..(i + h) {
                    let x = v[j];
                    let y = v[j + h];
                    v[j] = x + y;
                    v[j + h] = x - y;
                }
            }
            h *= 2;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rotator_preserves_norm() {
        let rotator = Rotator::new(64);
        
        // Random vector
        let x: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
        let norm_before: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        
        let y = rotator.rotate(&x);
        let norm_after: f32 = y.iter().map(|v| v * v).sum::<f32>().sqrt();
        
        // Norms should be approximately equal
        assert!((norm_before - norm_after).abs() < 0.01, 
                "Norms differ: {} vs {}", norm_before, norm_after);
    }

    #[test]
    fn test_rotation_roundtrip() {
        let rotator = Rotator::new(64);
        
        let x: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
        let y = rotator.rotate(&x);
        let z = rotator.rotate_inverse(&y);
        
        // Should be approximately equal to original
        for (a, b) in x.iter().zip(z.iter()) {
            assert!((a - b).abs() < 0.01, "Mismatch: {} vs {}", a, b);
        }
    }

    #[test]
    fn test_hadamard_basic() {
        let rotator = Rotator::new(4);
        
        // Test with simple input
        let x = vec![1.0, 0.0, 0.0, 0.0];
        let y = rotator.rotate(&x);
        
        // After Hadamard (with normalization), all components should have equal magnitude
        assert!(y.iter().all(|&v| (v.abs() - y[0].abs()).abs() < 0.01));
    }

    #[test]
    fn test_block_rotator() {
        let rotator = BlockRotator::new(768, 64);
        
        let x: Vec<f32> = (0..768).map(|i| (i as f32 - 384.0) * 0.01).collect();
        let norm_before: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        
        let y = rotator.rotate(&x);
        let norm_after: f32 = y.iter().map(|v| v * v).sum::<f32>().sqrt();
        
        assert!((norm_before - norm_after).abs() < 0.1);
    }
}
