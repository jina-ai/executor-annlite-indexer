from jina import Executor, requests
from typing import Optional, Dict
from docarray import Document, DocumentArray
from jina.logging.logger import JinaLogger


class AnnliteIndexer(Executor):
    def __init__(
        self,
        dim: int = 0,
        metric: str = 'cosine',
        limit: int = 10,
        ef_construction: int = 200,
        ef_query: int = 50,
        max_connection: int = 16,
        include_metadata: bool = True,
        index_traversal_paths: str = '@r',
        search_traversal_paths: str = '@r',
        columns: Optional[List[Tuple[str, str]]] = None,
        serialize_config: Optional[Dict] = None,
        *args,
        **kwargs,
    ):
        """
        :param dim: Dimensionality of vectors to index
        :param metric: Distance metric type. Can be 'euclidean', 'inner_product', or 'cosine'
        :param include_metadata: If True, return the document metadata in response
        :param limit: Number of results to get for each query document in search
        :param ef_construction: The construction time/accuracy trade-off
        :param ef_query: The query time accuracy/speed trade-off
        :param max_connection: The maximum number of outgoing connections in the
            graph (the "M" parameter)
        :param index_traversal_paths: Default traversal paths on docs
                (used for indexing, delete and update), e.g. '@r', '@c', '@r,c'
        :param search_traversal_paths: Default traversal paths on docs
        (used for search), e.g. '@r', '@c', '@r,c'
        :param columns: List of tuples of the form (column_name, str_type). Here str_type must be a string that can be
                parsed as a valid Python type.
        :param serialize_config: The configurations used for serializing documents, e.g., {'protocol': 'pickle'}
        """
        super().__init__(*args, **kwargs)
        self.logger = JinaLogger(self.__class__.__name__)

        assert dim > 0, 'Please specify the dimension of the vectors to index!'

        self.metric = metric
        self.limit = limit
        self.include_metadata = include_metadata
        self.index_traversal_paths = index_traversal_paths
        self.search_traversal_paths = search_traversal_paths
        self._valid_input_columns = ['str', 'float', 'int']

        if columns:
            cols = []
            for n, t in columns:
                assert (
                    t in self._valid_input_columns
                ), f'column of type={t} is not supported. Supported types are {self._valid_input_columns}'
                cols.append((n, eval(t)))
            columns = cols

        super().__init__(**kwargs)

        self._index = DocumentArray(
            storage='annlite',
            config={
                'n_dim': n_dim,
                'distance': distance,
                'ef_construct': ef_construct,
                'max_connection': max_connection,
                'data_path'= './workspace'
            },
        )


        self.logger = JinaLogger(self.metas.name)

    @requests(on='/index')
    def index(self, docs: DocumentArray, **kwargs):

        if docs:
            self._index.extend(docs)

    @requests(on='/search')
    def search(
        self,
        docs: 'DocumentArray',
        parameters: Optional[Dict] = None,
        **kwargs,
    ):
        """
        Perform a vector similarity search and retrieve the full Document match
        :param docs: the Documents to search with
        :param parameters: the runtime arguments to `DocumentArray`'s match
        function. They overwrite the original match_args arguments.
        """
        docs.match(self._index)

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

    @requests(on='/clear')
    def clear(self, **kwargs):
        """
        clear the database
        """
        self._index.clear()
