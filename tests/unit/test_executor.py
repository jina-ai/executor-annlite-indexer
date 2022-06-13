import operator

import pytest
from docarray.array.annlite import DocumentArrayAnnlite
from docarray import Document, DocumentArray

import numpy as np
from executor import AnnliteIndexer


def assert_document_arrays_equal(arr1, arr2):
    assert len(arr1) == len(arr2)
    for d1, d2 in zip(arr1, arr2):
        assert d1.id == d2.id
        assert d1.content == d2.content
        assert d1.chunks == d2.chunks
        assert d1.matches == d2.matches


@pytest.fixture
def docs():
    return DocumentArray(
        [
            Document(id='doc1', embedding=np.random.rand(128)),
            Document(id='doc2', embedding=np.random.rand(128)),
            Document(id='doc3', embedding=np.random.rand(128)),
            Document(id='doc4', embedding=np.random.rand(128)),
            Document(id='doc5', embedding=np.random.rand(128)),
            Document(id='doc6', embedding=np.random.rand(128)),
        ]
    )


@pytest.fixture
def update_docs():
    return DocumentArray(
        [
            Document(id='doc1', text='modified', embedding=np.random.rand(128)),
        ]
    )


def test_init(tmpdir):
    annlite_index = AnnliteIndexer(data_path=str(tmpdir), metric='euclidean', n_dim=10)

    assert isinstance(annlite_index._index, DocumentArrayAnnlite)
    assert annlite_index._index._config.metric == 'euclidean'
    assert annlite_index._index._config.n_dim == 10


def test_index(docs, tmpdir):
    annlite_index = AnnliteIndexer(data_path=str(tmpdir), metric='euclidean')
    annlite_index.index(docs)
    assert len(annlite_index._index) == len(docs)


def test_delete(docs, tmpdir):
    annlite_index = AnnliteIndexer(data_path=str(tmpdir), metric='euclidean')
    annlite_index.index(docs)

    ids = ['doc1', 'doc2', 'doc3']
    annlite_index.delete({'ids': ids})
    assert len(annlite_index._index) == len(docs) - 3
    for doc_id in ids:
        assert doc_id not in annlite_index._index


def test_update(docs, update_docs, tmpdir):
    # index docs first
    annlite_index = AnnliteIndexer(data_path=str(tmpdir), metric='euclidean')
    annlite_index.index(docs)
    assert_document_arrays_equal(annlite_index._index, docs)

    # update first doc
    annlite_index.update(update_docs)
    assert annlite_index._index[0].id == 'doc1'
    assert annlite_index._index['doc1'].text == 'modified'


def test_fill_embeddings(tmpdir):
    annlite_index = AnnliteIndexer(data_path=str(tmpdir), metric='euclidean', n_dim=1)

    annlite_index.index(DocumentArray([Document(id='a', embedding=np.array([1]))]))
    search_docs = DocumentArray([Document(id='a')])
    annlite_index.fill_embedding(search_docs)
    assert search_docs['a'].embedding is not None
    assert (search_docs['a'].embedding == np.array([1])).all()

    with pytest.raises(KeyError, match='b'):
        annlite_index.fill_embedding(DocumentArray([Document(id='b')]))


def test_persistence(docs, tmpdir):
    data_path = str(tmpdir)

    annlite_index1 = AnnliteIndexer(metric='euclidean', data_path=data_path)
    annlite_index1.index(docs)
    annlite_index2 = AnnliteIndexer(metric='euclidean', data_path=data_path)
    assert_document_arrays_equal(annlite_index2._index, docs)


@pytest.mark.parametrize(
    'metric, metric_name',
    [('euclidean', 'euclid_similarity'), ('cosine', 'cosine_similarity')],
)
def test_search(metric, metric_name, docs, tmpdir):
    # test general/normal case
    indexer = AnnliteIndexer(data_path=str(tmpdir), metric=metric)
    indexer.index(docs)
    query = DocumentArray([Document(embedding=np.random.rand(128)) for _ in range(10)])
    indexer.search(query)

    for doc in query:
        similarities = [t[metric_name].value for t in doc.matches[:, 'scores']]
        assert sorted(similarities, reverse=True) == similarities


def test_clear(docs, docker_compose, tmpdir):
    indexer = AnnliteIndexer(data_path=str(tmpdir))
    indexer.index(docs)
    assert len(indexer._index) == 6
    indexer.clear()
    assert len(indexer._index) == 0


@pytest.mark.parametrize('type_', ['int', 'float'])
def test_columns(docker_compose, tmpdir, type_):
    n_dim = 3
    indexer = AnnliteIndexer(
        data_path=str(tmpdir), n_dim=n_dim, columns=[('price', type_)]
    )

    docs = DocumentArray(
        [
            Document(id=f'r{i}', embedding=i * np.ones(n_dim), tags={'price': i})
            for i in range(10)
        ]
    )
    indexer.index(docs)
    assert len(indexer._index) == 10


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

    indexer = AnnliteIndexer(
        data_path=str(tmpdir), n_dim=n_dim, columns=[('price', 'float')]
    )

    docs = DocumentArray(
        [
            Document(id=f'r{i}', embedding=np.random.rand(n_dim), tags={'price': i})
            for i in range(50)
        ]
    )
    indexer.index(docs)

    for threshold in [10, 20, 30]:

        filter_ = {'price': {operator: threshold}}

        doc_query = DocumentArray([Document(embedding=np.random.rand(n_dim))])
        indexer.search(doc_query, parameters={'filter': filter_})

        assert len(doc_query[0].matches)

        assert all(
            [
                numeric_operators_annlite[operator](r.tags['price'], threshold)
                for r in doc_query[0].matches
            ]
        )
