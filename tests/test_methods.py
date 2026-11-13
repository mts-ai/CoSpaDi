import pytest
from paperflow.methods import MethodSpec


def test_only_ksvd_is_supported():
    assert MethodSpec('ksvd', 60).name == 'ksvd'


@pytest.mark.parametrize('method', ('ksvd_opp', 'ksvd_opp_weighted', 'svd', 'invalid'))
def test_other_methods_are_not_in_cospadi(method):
    with pytest.raises(ValueError):
        MethodSpec(method, 60)


@pytest.mark.parametrize('iterations,power,ratio', [(0, 8, 2), (60, 0, 2), (60, 8, 0)])
def test_invalid_solver_parameters(iterations, power, ratio):
    with pytest.raises(ValueError):
        MethodSpec('ksvd', iterations, power_iterations=power, ks_ratio=ratio)


