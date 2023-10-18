def test_cuda():
    from torch.cuda import is_available
    assert is_available()
    