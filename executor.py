from jina import Executor, requests
from typing import Optional, Dict, List, Tuple
from docarray import Document, DocumentArray
from jina.logging.logger import JinaLogger


class AnnliteIndexer(Executor):
    def __init__(
        self,
        n_dim: int = 128,
        metric: str = 'cosine',
        ef_construction: Optional[int] = None,
        ef_search: Optional[int] = None,
        max_connection: Optional[int] = None,
        columns: Optional[List[Tuple[str, str]]] = None,
        *args,
        **kwargs,
    ):
        """
        :param n_dim: Dimensionality of vectors to index
        :param metric: Distance metric type. Can be 'euclidean', 'inner_product', or 'cosine'
        :param max_connection: The maximum number of outgoing connections in the graph (the "M" parameter)
        :param include_metadata: If True, return the document metadata in response
        :param ef_construction: The construction time/accuracy trade-off
        :param ef_search: The query time accuracy/speed trade-off
        :param index_traversal_paths: Default traversal paths on docs
                (used for indexing, delete and update), e.g. '@r', '@c', '@r,c'
        :param columns: precise columns for the Indexer (used for filtering).
        """
        super().__init__(*args, **kwargs)
        self.logger = JinaLogger(self.__class__.__name__)
        super().__init__(**kwargs)

        config = {
            'n_dim': n_dim,
            'metric': metric,
            'ef_construction': ef_construction,
            'ef_search': ef_search,
            'max_connection': max_connection,
            'data_path': self.workspace or './workspace',
            'columns': columns,
        }

        self._index = DocumentArray(storage='annlite', config=config)
        self.logger = JinaLogger(self.metas.name)

    @requests(on='/index')
    def index(self, docs: DocumentArray, **kwargs):

        if docs:
            self._index.extend(docs)

    @requests(on='/search')
    def search(
        self,
        docs: 'DocumentArray',
        parameters: Dict = {},
        **kwargs,
    ):
        """Perform a vector similarity search and retrieve the full Document match

        :param docs: the Documents to search with
        :param parameters: Dictionary to define the `filter` that you want to use.
        :param kwargs: additional kwargs for the endpoint

        """
        docs.match(self._index, filter=parameters.get('filter', None))

    @requests(on='/delete')
    def delete(self, parameters: Dict, **kwargs):
        """
        Delete entries from the index by id
        :param parameters: parameters to the request
        """
        deleted_ids = parameters.get('ids', [])
        if len(deleted_ids) == 0:
            return

        del self._index[deleted_ids]

    @requests(on='/update')
    def update(self, docs: DocumentArray, **kwargs):
        """
        Update doc with the same id, if not present, append into storage
        :param docs: the documents to update
        """

        for doc in docs:
            try:
                self._index[doc.id] = doc
            except IndexError:
                self.logger.warning(
                    f'cannot update doc {doc.id} as it does not exist in storage'
                )

    @requests(on='/fill_embedding')
    def fill_embedding(self, docs: DocumentArray, **kwargs):
        """
        retrieve embedding of Documents by id
        :param docs: DocumentArray to search with
        """
        for doc in docs:
            doc.embedding = self._index[doc.id].embedding

    @requests(on='/filter')
    def filter(self, parameters: Dict, **kwargs):
        """
        Query documents from the indexer by the filter `query` object in parameters. The `query` object must follow the
        specifications in the `find` method of `DocumentArray` using annlite: https://docarray.jina.ai/fundamentals/documentarray/find/#filter-with-query-operators
        :param parameters: Dictionary to define the `filter` that you want to use.
        """
        return self._index.find(parameters.get('filter', None))

    @requests(on='/clear')
    def clear(self, **kwargs):
        """
        clear the database
        """
        self._index.clear()

    def close(self) -> None:
        super().close()
        del self._index
