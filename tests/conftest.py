import os
import time

import pytest

cur_dir = os.path.dirname(os.path.abspath(__file__))
compose_yml = os.path.abspath(os.path.join(cur_dir, 'Dockerfile'))

@pytest.fixture(scope='function')
def docker_compose():
    os.system(
        f"docker start -f {compose_yml} --project-directory . up  --build -d --remove-orphans"
    )
    time.sleep(5)
    yield
    os.system(
        f"docker start -f {compose_yml} --project-directory . down --remove-orphans"
    )
