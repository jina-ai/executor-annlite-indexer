from jina import Executor, requests
from typing import Optional, Dict, List, Tuple
from docarray import Document, DocumentArray
from jina.logging.logger import JinaLogger


class AnnLiteIndexer(Executor):
    def __init__(
        self,
        n_dim: int = 128,
        metric: str = 'cosine',
        limit: int = 10,
        data_path: Optional[str] = None,
        ef_construction: Optional[int] = None,
        ef_search: Optional[int] = None,
        max_connection: Optional[int] = None,
        include_metadata: bool = True,
        index_traversal_paths: str = '@r',
        search_traversal_paths: str = '@r',
        columns: Optional[List[Tuple[str, str]]] = None,
        *args,
        **kwargs,
    ):
        """
        :param n_dim: Dimensionality of vectors to index
        :param metric: Distance metric type. Can be 'euclidean', 'inner_product', or 'cosine'
        :param limit: Number of results to get for each query document in search
        :param data_path: Path of the folder where to store indexed data.
        :param max_connection: The maximum number of outgoing connections in the graph (the "M" parameter)
        :param include_metadata: If True, return the document metadata in response
        :param ef_construction: The construction time/accuracy trade-off
        :param ef_search: The query time accuracy/speed trade-off
        :param index_traversal_paths: Default traversal paths on docs
                (used for indexing, delete and update), e.g. '@r', '@c', '@r,c'
        :param search_traversal_paths: Default traversal paths on docs
        (used for search), e.g. '@r', '@c', '@r,c'
        :param columns: precise columns for the Indexer (used for filtering).
        """
        super().__init__(*args, **kwargs)
        self.logger = JinaLogger(self.__class__.__name__)
        self.limit = limit
        self.include_metadata = include_metadata
        self.index_traversal_paths = index_traversal_paths
        self.search_traversal_paths = search_traversal_paths

        config = {
            'n_dim': n_dim,
            'metric': metric,
            'ef_construction': ef_construction,
            'ef_search': ef_search,
            'max_connection': max_connection,
            'data_path': data_path or self.workspace or './workspace',
            'columns': columns,
        }

        self._index = DocumentArray(storage='annlite', config=config)

    @requests(on='/index')
    def index(self, docs: DocumentArray, parameters: dict = {}, **kwargs):
        """Index new documents
        :param docs: the Documents to index
        :param parameters: dictionary with options for indexing
        Keys accepted:
            - 'traversal_paths' (str): traversal path for the docs
        """
        traversal_paths = parameters.get('traversal_paths', self.index_traversal_paths)
        flat_docs = docs[traversal_paths]
        if len(flat_docs) == 0:
            return

        self._index.extend(flat_docs)

    @requests(on='/search')
    def search(
        self,
        docs: 'DocumentArray',
        parameters: Dict = {},
        **kwargs,
    ):
        """Perform a vector similarity search and retrieve the full Document match

        :param docs: the Documents to search with
        :param parameters: dictionary for parameters for the search operation
        Keys accepted:
            - 'filter' (dict): the filtering conditions on document tags
            - 'traversal_paths' (str): traversal paths for the docs
            - 'limit' (int): nr of matches to get per Document
        :param kwargs: additional kwargs for the endpoint

        """
        if not docs:
            return

        limit = int(parameters.get('limit', self.limit))
        docs.match(self._index, filter=parameters.get('filter', None), limit=limit)

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
    def update(self, docs: DocumentArray, parameters: dict = {}, **kwargs):
        """Update existing documents
        :param docs: the Documents to update
        :param parameters: dictionary with options for updating
        Keys accepted:
            - 'traversal_paths' (str): traversal path for the docs
        """
        if not docs:
            return

        traversal_paths = parameters.get('traversal_paths', self.index_traversal_paths)
        flat_docs = docs[traversal_paths]
        if len(flat_docs) == 0:
            return

        for doc in flat_docs:
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

    @requests(on='/status')
    def status(self, **kwargs) -> DocumentArray:
        """Return the document containing status information about the indexer.
        The status will contain information on the total number of indexed and deleted
        documents, and on the number of (searchable) documents currently in the index.
        """

        status = Document(tags=self._index._annlite.stat)
        return DocumentArray([status])

    @requests(on='/clear')
    def clear(self, **kwargs):
        """
        clear the database
        """
        self._index.clear()

    def close(self) -> None:
        super().close()
        del self._index
