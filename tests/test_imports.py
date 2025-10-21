def test_imports():
    import xadl
    from xadl.models.encoder import MLPEncoder, CNN1DEncoder
    assert MLPEncoder and CNN1DEncoder
