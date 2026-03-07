import pickle
import quadrants as qd
from tests import test_utils
from pathlib import Path


@test_utils.test()
def test_pickle_ndarray(tmp_path: Path) -> None:
    a = qd.ndarray(qd.i32, (3, 2))

    a[0, 1] = 3
    a[1, 1] = 5

    with open(tmp_path / "foo.pkl", "wb") as f:
        pickle.dump(a, f)

    with open(tmp_path / "foo.pkl", "rb") as f:
        b = pickle.load(f)

    assert b[0, 1] == 3
    assert b[1, 1] == 5
    assert b.shape == a.shape
    assert b.dtype == a.dtype
