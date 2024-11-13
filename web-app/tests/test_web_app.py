"""Used to test web app component."""
import pytest

class Tests:
    """Test functions"""
    def test_sanity_check(self):
        """
        Test debugging... making sure that we can run a simple test that always passes.
        Note the use of the example_fixture in the parameter list - 
        any setup and teardown in that fixture will be run before 
        and after this test function executes.
        """
        expected = True
        actual = True
        assert actual == expected, "Expected True to be equal to True!"
        
    #write the rest of the tests here.