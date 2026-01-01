import { Query } from './query';

// Mock IpcClient for testing
const mockClient = {
  query: jest.fn(),
};

describe('Query', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('builder pattern', () => {
    it('should chain limit()', () => {
      const query = new Query(mockClient as any, 'users/');
      const result = query.limit(10);
      expect(result).toBe(query);
    });

    it('should chain offset()', () => {
      const query = new Query(mockClient as any, 'users/');
      const result = query.offset(5);
      expect(result).toBe(query);
    });

    it('should chain select()', () => {
      const query = new Query(mockClient as any, 'users/');
      const result = query.select(['name', 'email']);
      expect(result).toBe(query);
    });

    it('should chain multiple methods', () => {
      const query = new Query(mockClient as any, 'users/')
        .limit(10)
        .offset(5)
        .select(['name', 'email']);
      expect(query).toBeDefined();
    });
  });

  describe('execute()', () => {
    it('should call client.query with correct parameters', async () => {
      mockClient.query.mockResolvedValue('result[0]{}:');
      
      const query = new Query(mockClient as any, 'users/')
        .limit(10)
        .offset(5)
        .select(['name', 'email']);
      
      await query.execute();
      
      expect(mockClient.query).toHaveBeenCalledWith('users/', {
        limit: 10,
        offset: 5,
        columns: ['name', 'email'],
      });
    });
  });

  describe('toList()', () => {
    it('should parse empty TOON result', async () => {
      mockClient.query.mockResolvedValue('result[0]{}:');
      
      const query = new Query(mockClient as any, 'users/');
      const results = await query.toList();
      
      expect(results).toEqual([]);
    });

    it('should parse TOON result with data', async () => {
      mockClient.query.mockResolvedValue('result[2]{name,age}: Alice,30; Bob,25');
      
      const query = new Query(mockClient as any, 'users/');
      const results = await query.toList();
      
      expect(results).toEqual([
        { name: 'Alice', age: 30 },
        { name: 'Bob', age: 25 },
      ]);
    });

    it('should parse JSON fallback', async () => {
      mockClient.query.mockResolvedValue('[{"name":"Alice"},{"name":"Bob"}]');
      
      const query = new Query(mockClient as any, 'users/');
      const results = await query.toList();
      
      expect(results).toEqual([
        { name: 'Alice' },
        { name: 'Bob' },
      ]);
    });

    it('should parse boolean and null values', async () => {
      mockClient.query.mockResolvedValue('result[1]{active,deleted}: true,null');
      
      const query = new Query(mockClient as any, 'users/');
      const results = await query.toList();
      
      expect(results).toEqual([
        { active: true, deleted: null },
      ]);
    });
  });

  describe('first()', () => {
    it('should return first result', async () => {
      mockClient.query.mockResolvedValue('result[1]{name}: Alice');
      
      const query = new Query(mockClient as any, 'users/');
      const result = await query.first();
      
      expect(result).toEqual({ name: 'Alice' });
    });

    it('should return null for empty result', async () => {
      mockClient.query.mockResolvedValue('result[0]{}:');
      
      const query = new Query(mockClient as any, 'users/');
      const result = await query.first();
      
      expect(result).toBeNull();
    });
  });

  describe('count()', () => {
    it('should return count of results', async () => {
      mockClient.query.mockResolvedValue('result[3]{name}: Alice; Bob; Charlie');
      
      const query = new Query(mockClient as any, 'users/');
      const count = await query.count();
      
      expect(count).toBe(3);
    });
  });
});
