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


def test_init():
    annlite_index = AnnliteIndexer(metric='euclidean', n_dim=10)

    assert isinstance(annlite_index._index, DocumentArrayAnnlite)
    assert annlite_index._index._config.metric == 'euclidean'
    assert annlite_index._index._config.n_dim == 10


def test_index(docs):
    annlite_index = AnnliteIndexer(metric='euclidean', n_dim=128)
    annlite_index.index(docs)
    assert len(annlite_index._index) == len(docs)


def test_delete(docs):
    annlite_index = AnnliteIndexer(metric='euclidean', n_dim=128)
    annlite_index.index(docs)

    ids = ['doc1', 'doc2', 'doc3']
    annlite_index.delete({'ids': ids})
    assert len(annlite_index._index) == len(docs) - 3
    for doc_id in ids:
        assert doc_id not in annlite_index._index


def test_update(docs, update_docs):
    # index docs first
    annlite_index = AnnliteIndexer(metric='euclidean', n_dim=128)
    annlite_index.index(docs)
    assert_document_arrays_equal(annlite_index._index, docs)

    # update first doc
    annlite_index.update(update_docs)
    assert annlite_index._index[0].id == 'doc1'
    assert annlite_index._index['doc1'].text == 'modified'


def test_fill_embeddings():
    annlite_index = AnnliteIndexer(metric='euclidean', n_dim=1)

    annlite_index.index(DocumentArray([Document(id='a', embedding=np.array([1]))]))
    search_docs = DocumentArray([Document(id='a')])
    annlite_index.fill_embedding(search_docs)
    assert search_docs['a'].embedding is not None
    assert (search_docs['a'].embedding == np.array([1])).all()

    with pytest.raises(KeyError, match='b'):
        annlite_index.fill_embedding(DocumentArray([Document(id='b')]))


def test_persistence(docs):
    from tempfile import TemporaryDirectory
    data_path = TemporaryDirectory().name


    annlite_index1 = AnnliteIndexer(metric='euclidean', n_dim=128, data_path=data_path)
    annlite_index1.index(docs)
    annlite_index2 = AnnliteIndexer(metric='euclidean', n_dim=128, data_path=data_path)
    assert_document_arrays_equal(annlite_index2._index, docs)


@pytest.mark.parametrize(
    'metric, metric_name',
    [('euclidean', 'euclid_similarity'), ('cosine', 'cosine_similarity')],
)
def test_search(metric, metric_name, docs):
    # test general/normal case
    indexer = AnnliteIndexer(metric=metric, n_dim=128)
    indexer.index(docs)
    query = DocumentArray([Document(embedding=np.random.rand(128)) for _ in range(10)])
    indexer.search(query, {})

    for doc in query:
        similarities = [t[metric_name].value for t in doc.matches[:, 'scores']]
        assert sorted(similarities, reverse=True) == similarities
