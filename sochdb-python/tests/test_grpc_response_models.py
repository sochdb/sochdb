from sochdb.proto import sochdb_pb2


def test_search_response_exposes_metric_without_breaking_results_access():
    response = sochdb_pb2.SearchResponse(
        results=[
            sochdb_pb2.SearchResult(
                id=42,
                distance=0.25,
                metric="cosine",
            )
        ],
        duration_us=99,
        metric=sochdb_pb2.DISTANCE_METRIC_COSINE,
    )

    encoded = response.SerializeToString()
    decoded = sochdb_pb2.SearchResponse.FromString(encoded)

    assert decoded.metric == sochdb_pb2.DISTANCE_METRIC_COSINE
    assert decoded.results[0].id == 42
    assert decoded.results[0].distance == 0.25
    assert decoded.results[0].metric == "cosine"


def test_legacy_search_response_defaults_metric_to_unspecified():
    legacy = sochdb_pb2.SearchResponse(
        results=[sochdb_pb2.SearchResult(id=7, distance=1.5)],
        duration_us=12,
    )

    encoded = legacy.SerializeToString()
    assert b"\x20" not in encoded

    decoded = sochdb_pb2.SearchResponse.FromString(encoded)
    assert decoded.metric == sochdb_pb2.DISTANCE_METRIC_UNSPECIFIED
    assert decoded.results[0].id == 7
    assert decoded.results[0].distance == 1.5


def test_search_batch_response_exposes_metric():
    response = sochdb_pb2.SearchBatchResponse(
        results=[
            sochdb_pb2.QueryResults(
                results=[
                    sochdb_pb2.SearchResult(
                        id=11,
                        distance=-3.0,
                        metric="dot_product",
                    )
                ]
            )
        ],
        duration_us=44,
        metric=sochdb_pb2.DISTANCE_METRIC_DOT_PRODUCT,
    )

    decoded = sochdb_pb2.SearchBatchResponse.FromString(response.SerializeToString())

    assert decoded.metric == sochdb_pb2.DISTANCE_METRIC_DOT_PRODUCT
    assert decoded.results[0].results[0].id == 11
    assert decoded.results[0].results[0].distance == -3.0


def test_search_result_metadata_presence_round_trips():
    response = sochdb_pb2.SearchResponse(
        results=[
            sochdb_pb2.SearchResult(
                id=1,
                distance=0.0,
                metric="cosine",
                parent_id=0,
                view_type="turn",
            ),
            sochdb_pb2.SearchResult(
                id=2,
                distance=1.0,
                metric="cosine",
            ),
        ],
        metric=sochdb_pb2.DISTANCE_METRIC_COSINE,
    )

    decoded = sochdb_pb2.SearchResponse.FromString(response.SerializeToString())

    assert decoded.results[0].HasField("parent_id")
    assert decoded.results[0].parent_id == 0
    assert decoded.results[0].HasField("view_type")
    assert decoded.results[0].view_type == "turn"
    assert not decoded.results[1].HasField("parent_id")
    assert not decoded.results[1].HasField("view_type")


def test_insert_batch_metadata_model_supports_mixed_presence():
    request = sochdb_pb2.InsertBatchRequest(
        index_name="vectors",
        ids=[1, 2],
        vectors=[
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
        ],
        metadata=[
            sochdb_pb2.VectorMetadata(parent_id=0, view_type="turn"),
            sochdb_pb2.VectorMetadata(),
        ],
    )

    decoded = sochdb_pb2.InsertBatchRequest.FromString(request.SerializeToString())

    assert decoded.metadata[0].HasField("parent_id")
    assert decoded.metadata[0].parent_id == 0
    assert decoded.metadata[0].view_type == "turn"
    assert not decoded.metadata[1].HasField("parent_id")
    assert not decoded.metadata[1].HasField("view_type")

