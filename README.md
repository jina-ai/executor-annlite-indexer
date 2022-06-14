
# AnnliteIndexer

AnnliteIndexer indexes Documents into a `DocumentArray`  using `storage='annlite'`. Underneath, the `DocumentArray`  uses 
 [AnnLite](https://github.com/jina-ai/annlite) to store and search Documents efficiently. 

## Usage

#### via Docker image (recommended)

```python
from jina import Flow
	
f = Flow().add(uses='jinahub+docker://AnnliteIndexer')
```

#### via source code

```python
from jina import Flow
	
f = Flow().add(uses='jinahub://AnnliteIndexer')
```

- To override `__init__` args & kwargs, use `.add(..., uses_with: {'key': 'value'})`
- To override class metas, use `.add(..., uses_metas: {'key': 'value})`


## Vector Search

The following example shows how to perform vector search using`f.post(on='/search', inputs=[Document(embedding=np.array([1,1]))])`.


```python
from jina import Flow
from docarray import Document
import numpy as np

f = Flow().add(
         uses='jinahub://AnnliteIndexer',
         uses_with={'n_dim': 2},
     )

with f:
    f.post(
        on='/index',
        inputs=[
            Document(id='a', embedding=np.array([1, 3])),
            Document(id='b', embedding=np.array([1, 1])),
        ],
    )

    docs = f.post(
        on='/search',
        inputs=[Document(embedding=np.array([1, 1]))],
    )

# will print "The ID of the best match of [1,1] is: b"
print('The ID of the best match of [1,1] is: ', docs[0].matches[0].id)
```


### Using filtering
To do filtering with the AnnliteIndexer you should first define columns and precise the dimension of your embedding space.
For instance :


```python
from jina import Flow

f = Flow().add(
    uses='jinahub+docker://AnnliteIndexer',
    uses_with={
        'data_path': 'data_path/',
        'n_dim': 256,
        'columns': [('price', 'float')],
    },
)

```

Then you can pass a filter as a parameters when searching for document:
```python
from docarray import Document, DocumentArray
import numpy as np

docs = DocumentArray(
    [
        Document(id=f'r{i}', embedding=np.random.rand(3), tags={'price': i})
        for i in range(50)
    ]
)

filter_ = {'price': {'$lte': 30}}

with f:
    f.index(docs)
    doc_query = DocumentArray([Document(embedding=np.random.rand(3))])
    f.search(doc_query, parameters={'filter': filter_})
```

For more information please refer to the docarray [documentation](https://docarray.jina.ai/advanced/document-store/annlite/#vector-search-with-filter)


## tests


Test can be run setting the `PYTHONPATH` into the root of this repository
```
export PYTHONPATH=$PYTHONPATH:`pwd`
```
and then running

```
pytest tests
```