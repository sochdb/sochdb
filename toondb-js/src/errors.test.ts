import {
  ToonDBError,
  ConnectionError,
  TransactionError,
  ProtocolError,
  DatabaseError,
} from './errors';

describe('ToonDB Errors', () => {
  describe('ToonDBError', () => {
    it('should be an instance of Error', () => {
      const error = new ToonDBError('test message');
      expect(error).toBeInstanceOf(Error);
      expect(error).toBeInstanceOf(ToonDBError);
      expect(error.name).toBe('ToonDBError');
      expect(error.message).toBe('test message');
    });
  });

  describe('ConnectionError', () => {
    it('should be an instance of ToonDBError', () => {
      const error = new ConnectionError('connection failed');
      expect(error).toBeInstanceOf(ToonDBError);
      expect(error).toBeInstanceOf(ConnectionError);
      expect(error.name).toBe('ConnectionError');
      expect(error.message).toBe('connection failed');
    });
  });

  describe('TransactionError', () => {
    it('should be an instance of ToonDBError', () => {
      const error = new TransactionError('transaction failed');
      expect(error).toBeInstanceOf(ToonDBError);
      expect(error).toBeInstanceOf(TransactionError);
      expect(error.name).toBe('TransactionError');
      expect(error.message).toBe('transaction failed');
    });
  });

  describe('ProtocolError', () => {
    it('should be an instance of ToonDBError', () => {
      const error = new ProtocolError('protocol error');
      expect(error).toBeInstanceOf(ToonDBError);
      expect(error).toBeInstanceOf(ProtocolError);
      expect(error.name).toBe('ProtocolError');
      expect(error.message).toBe('protocol error');
    });
  });

  describe('DatabaseError', () => {
    it('should be an instance of ToonDBError', () => {
      const error = new DatabaseError('database error');
      expect(error).toBeInstanceOf(ToonDBError);
      expect(error).toBeInstanceOf(DatabaseError);
      expect(error.name).toBe('DatabaseError');
      expect(error.message).toBe('database error');
    });
  });
});
