/**
 * Tests for VectorIndex
 */

import { VectorIndex } from './vector';

describe('VectorIndex', () => {
  describe('distance calculations', () => {
    describe('computeCosineDistance', () => {
      it('should return 0 for identical vectors', () => {
        const v = [0.5, 0.5, 0.5];
        expect(VectorIndex.computeCosineDistance(v, v)).toBeCloseTo(0, 5);
      });

      it('should return 2 for opposite vectors', () => {
        const a = [1, 0, 0];
        const b = [-1, 0, 0];
        expect(VectorIndex.computeCosineDistance(a, b)).toBeCloseTo(2, 5);
      });

      it('should return 1 for orthogonal vectors', () => {
        const a = [1, 0, 0];
        const b = [0, 1, 0];
        expect(VectorIndex.computeCosineDistance(a, b)).toBeCloseTo(1, 5);
      });

      it('should work with high-dimensional vectors', () => {
        const dim = 384;
        const a = new Array(dim).fill(0).map((_, i) => Math.sin(i));
        const b = new Array(dim).fill(0).map((_, i) => Math.sin(i));
        expect(VectorIndex.computeCosineDistance(a, b)).toBeCloseTo(0, 5);
      });
    });

    describe('computeEuclideanDistance', () => {
      it('should return 0 for identical vectors', () => {
        const v = [1, 2, 3];
        expect(VectorIndex.computeEuclideanDistance(v, v)).toBeCloseTo(0, 5);
      });

      it('should compute 3-4-5 triangle correctly', () => {
        const a = [0, 0];
        const b = [3, 4];
        expect(VectorIndex.computeEuclideanDistance(a, b)).toBeCloseTo(5, 5);
      });

      it('should work with negative values', () => {
        const a = [-1, -1];
        const b = [1, 1];
        expect(VectorIndex.computeEuclideanDistance(a, b)).toBeCloseTo(2 * Math.sqrt(2), 5);
      });
    });

    describe('computeDotProduct', () => {
      it('should compute dot product correctly', () => {
        const a = [1, 2, 3];
        const b = [4, 5, 6];
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        expect(VectorIndex.computeDotProduct(a, b)).toBeCloseTo(32, 5);
      });

      it('should return 0 for orthogonal vectors', () => {
        const a = [1, 0];
        const b = [0, 1];
        expect(VectorIndex.computeDotProduct(a, b)).toBeCloseTo(0, 5);
      });
    });
  });

  describe('normalizeVector', () => {
    it('should normalize to unit length', () => {
      const v = [3, 4];
      const n = VectorIndex.normalizeVector(v);
      
      const length = Math.sqrt(n[0] * n[0] + n[1] * n[1]);
      expect(length).toBeCloseTo(1, 5);
    });

    it('should preserve direction', () => {
      const v = [6, 8];
      const n = VectorIndex.normalizeVector(v);
      
      expect(n[0]).toBeCloseTo(0.6, 5);
      expect(n[1]).toBeCloseTo(0.8, 5);
    });

    it('should handle zero vector', () => {
      const v = [0, 0, 0];
      const n = VectorIndex.normalizeVector(v);
      expect(n).toEqual([0, 0, 0]);
    });

    it('should handle single-element vector', () => {
      const v = [5];
      const n = VectorIndex.normalizeVector(v);
      expect(n[0]).toBeCloseTo(1, 5);
    });

    it('should not modify original vector', () => {
      const v = [3, 4];
      VectorIndex.normalizeVector(v);
      expect(v).toEqual([3, 4]);
    });
  });

  describe('VectorIndex construction', () => {
    it('should create with path only', () => {
      const index = new VectorIndex('./test_vectors');
      expect(index).toBeDefined();
    });

    it('should create with full config', () => {
      const index = new VectorIndex('./test_vectors', {
        dimension: 768,
        metric: 'euclidean',
        m: 32,
        efConstruction: 200,
      });
      expect(index).toBeDefined();
    });

    it('should use default values for missing config', () => {
      const index = new VectorIndex('./test_vectors', {
        dimension: 384,
      });
      expect(index).toBeDefined();
    });
  });

  describe('bulkBuild validation', () => {
    it('should validate empty vectors', async () => {
      const index = new VectorIndex('./test_vectors');
      
      // Empty array should not throw
      await expect(index.bulkBuild([])).resolves.not.toThrow();
    });

    it('should validate dimension consistency', async () => {
      const index = new VectorIndex('./test_vectors');
      
      const vectors = [
        [1, 2, 3],
        [1, 2], // Different dimension!
      ];
      
      await expect(index.bulkBuild(vectors)).rejects.toThrow(/dimension/i);
    });
  });
});
