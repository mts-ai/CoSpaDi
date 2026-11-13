import pickle
import pytest
torch=pytest.importorskip('torch')
from paperflow.statistics import GramAccumulator,sample_token_blocks
def test_gram_matches_explicit_xtx(tmp_path):
    x=torch.tensor([[[1.,2.],[3.,4.]]]);acc=GramAccumulator();acc.add(x)
    acc.add(x)
    torch.testing.assert_close(acc.value,2*(x.reshape(-1,2).T@x.reshape(-1,2)))
    assert acc.tokens==4
    p=tmp_path/'m.pkl';acc.save(p)
    with p.open('rb') as f:torch.testing.assert_close(pickle.load(f),acc.value)
def test_selection_deterministic():
    data=[[i]*12 for i in range(6)]
    assert sample_token_blocks(data,length=4,samples=3,seed=123)==sample_token_blocks(
        data,length=4,samples=3,seed=123)
def test_selection_errors():
    with pytest.raises(ValueError):sample_token_blocks([[1,2]],length=4,samples=1,seed=42)
