from jina import Executor, requests
from typing import Optional, Dict, List, Tuple
from docarray import Document, DocumentArray
from jina.logging.logger import JinaLogger


class AnnliteIndexer(Executor):
    def __init__(
        self,
        n_dim=2,
        metric: str = 'cosine',
        ef_construction: int = 200,
        ef_search: int = 50,
        max_connection: int = 16,
        data_path: Optional[str] = None,
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
        :param data_path: Path of the folder where to store indexed data.
        """
        super().__init__(*args, **kwargs)
        self.logger = JinaLogger(self.__class__.__name__)
        super().__init__(**kwargs)

        config = {'n_dim': n_dim,
                  'metric': metric,
                  'ef_construction': ef_construction,
                  'ef_search': ef_search,
                  'max_connection': max_connection,
                  'data_path': data_path}

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
        **kwargs,
    ):
        """
        Perform a vector similarity search and retrieve the full Document match
        :param docs: the Documents to search with
        :param parameters: the runtime arguments to `DocumentArray`'s match
        function. They overwrite the original match_args arguments.
        """
        docs.match(self._index)
        return docs

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
