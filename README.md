
# AnnliteIndexer

AnnliteIndexer indexes Documents into a DocumentArray  with ` `storage='annlite'`

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


#### tests


Test can be run setting the `PYTHONPATH` into the root of this repository
```
export PYTHONPATH=$PYTHONPATH:`pwd`
```
and then running

```
pytest tests
```