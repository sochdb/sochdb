/**
 * Tests for IpcClient wire protocol
 */

import { IpcClient, OpCode } from './ipc-client';

describe('IpcClient', () => {
  describe('OpCode', () => {
    it('should have correct operation codes', () => {
      expect(OpCode.Get).toBe(0x01);
      expect(OpCode.Put).toBe(0x02);
      expect(OpCode.Delete).toBe(0x03);
      expect(OpCode.GetPath).toBe(0x04);
      expect(OpCode.PutPath).toBe(0x05);
      expect(OpCode.Query).toBe(0x06);
      expect(OpCode.BeginTxn).toBe(0x10);
      expect(OpCode.CommitTxn).toBe(0x11);
      expect(OpCode.AbortTxn).toBe(0x12);
      expect(OpCode.Checkpoint).toBe(0x20);
      expect(OpCode.Stats).toBe(0x21);
      expect(OpCode.OK).toBe(0x80);
      expect(OpCode.Error).toBe(0x81);
      expect(OpCode.NotFound).toBe(0x82);
    });
  });

  describe('message encoding', () => {
    it('should encode key correctly', () => {
      const key = Buffer.from('test-key');
      const encoded = IpcClient.encodeKey(key);
      
      // Should be: [length:4][op:1][key_len:4][key:...]
      expect(encoded.length).toBe(4 + 1 + 4 + key.length);
    });

    it('should encode key-value correctly', () => {
      const key = Buffer.from('test-key');
      const value = Buffer.from('test-value');
      const encoded = IpcClient.encodeKeyValue(key, value);
      
      // Should be: [length:4][op:1][key_len:4][key:...][value_len:4][value:...]
      expect(encoded.length).toBe(4 + 1 + 4 + key.length + 4 + value.length);
    });
  });

  describe('static connect', () => {
    it('should throw ConnectionError for invalid socket', async () => {
      await expect(IpcClient.connect('/nonexistent/socket.sock')).rejects.toThrow();
    });
  });
});
