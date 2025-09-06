def test_always_passes():
    """Simple test that always passes"""
    assert True


def test_basic_math():
    """Basic math test"""
    assert 2 + 2 == 4
    assert 3 * 3 == 9


def test_string_operations():
    """Basic string operations test"""
    assert "hello" + " world" == "hello world"
    assert len("test") == 4
