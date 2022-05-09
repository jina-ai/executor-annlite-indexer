import numpy as np
import pytest
from docarray import Document, DocumentArray
from jina import Flow

from executor import AnnliteIndexer


def test_flow(tmpdir):
    f = Flow().add(
        uses=AnnliteIndexer,
        uses_with={'data_path': str(tmpdir), 'n_dim': 2, 'metric': 'euclidean'},
    )

    with f:
        f.post(
            on='/index',
            inputs=[
                Document(id='a', embedding=np.array([1, 3])),
                Document(id='b', embedding=np.array([1, 1])),
                Document(id='c', embedding=np.array([3, 1])),
                Document(id='d', embedding=np.array([2, 3])),
            ],
        )
        docs = f.post(on='/search', inputs=[Document(embedding=np.array([1, 1]))],)

    assert docs[0].matches[0].id == 'b'



def test_reload_keep_state(tmpdir):

    docs = DocumentArray([Document(embedding=np.random.rand(3)) for _ in range(2)])
    f = Flow().add(uses=AnnliteIndexer, uses_with={'data_path': str(tmpdir), 'n_dim': 3}, )

    with f:
        f.index(docs)
        first_search = f.search(inputs=docs)
        first_matches = first_search[0].matches

    with f:
        second_search = f.search(inputs=docs)
        second_matches = second_search[0].matches

    assert first_matches == second_matches