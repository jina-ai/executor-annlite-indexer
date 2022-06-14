import operator

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
        docs = f.post(
            on='/search',
            inputs=[Document(embedding=np.array([1, 1]))],
        )

    assert docs[0].matches[0].id == 'b'


def test_reload_keep_state(tmpdir):

    docs = DocumentArray([Document(embedding=np.random.rand(3)) for _ in range(2)])
    f = Flow().add(
        uses=AnnliteIndexer,
        uses_with={'data_path': str(tmpdir), 'n_dim': 3},
    )

    with f:
        f.index(docs)
        first_search = f.search(inputs=docs)
        first_matches = first_search[0].matches

    with f:
        second_search = f.search(inputs=docs)
        second_matches = second_search[0].matches

    assert first_matches == second_matches


def test_filter(tmpdir):
    n_dim = 3

    f = Flow().add(
        uses=AnnliteIndexer,
        uses_with={
            'data_path': str(tmpdir),
            'n_dim': n_dim,
            'columns': [('price', 'float')],
        },
    )

    docs = DocumentArray([Document(id=f'r{i}', tags={'price': i}) for i in range(10)])

    with f:
        f.index(docs)

        max_price = 3
        filter_ = {'price': {'$eq': max_price}}

        result = f.post(
            on='/filter', parameters={'filter': filter_}
        )

        assert len(result) == 1
        assert result[0].tags['price'] == max_price

numeric_operators_annlite = {
    '$gte': operator.ge,
    '$gt': operator.gt,
    '$lte': operator.le,
    '$lt': operator.lt,
    '$eq': operator.eq,
    '$neq': operator.ne,
}


@pytest.mark.parametrize('operator', list(numeric_operators_annlite.keys()))
def test_filtering(docker_compose, tmpdir, operator: str):
    n_dim = 256

    f = Flow().add(
        uses=AnnliteIndexer,
        uses_with={
            'data_path': str(tmpdir),
            'n_dim': n_dim,
            'columns': [('price', 'float')],
        },
    )

    docs = DocumentArray(
        [
            Document(id=f'r{i}', embedding=np.random.rand(n_dim), tags={'price': i})
            for i in range(50)
        ]
    )
    with f:

        f.index(docs)

        for threshold in [10, 20, 30]:
            filter_ = {'price': {operator: threshold}}

            doc_query = DocumentArray([Document(embedding=np.random.rand(n_dim))])
            indexed_docs = f.search(doc_query, parameters={'filter': filter_})

            assert len(indexed_docs[0].matches) > 0

            assert all(
                [
                    numeric_operators_annlite[operator](r.tags['price'], threshold)
                    for r in indexed_docs[0].matches
                ]
            )
