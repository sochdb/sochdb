import grpc
import sochdb_pb2, sochdb_pb2_grpc

channel = grpc.insecure_channel('localhost:50051')
stub = sochdb_pb2_grpc.VectorIndexServiceStub(channel)

# Create index
resp = stub.CreateIndex(sochdb_pb2.CreateIndexRequest(
    name="test",
    dimension=3
))
print("CreateIndex:", resp)

# Insert vectors
resp = stub.InsertBatch(sochdb_pb2.InsertBatchRequest(
    index_name="test",
    ids=[1, 2, 3],
    vectors=[0.0, 0.0, 0.0,  1.0, 2.0, 3.0,  4.0, 5.0, 6.0]
))
print("InsertBatch:", resp)


# Search
resp = stub.Search(sochdb_pb2.SearchRequest(
    index_name="test",
    query=[0.0, 0.0, 0.0],
    k=2
))
print("Search:", resp)
