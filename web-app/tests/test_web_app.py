"""Used to test web app component."""
import pytest
from unittest.mock import patch, Mock
from app import create_app


@pytest.fixture
def app():
    """Fixture for creating and configuring the Flask app."""
    app = create_app()
    return app

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

    def test_home_page(self, app):
        """Test the home page route."""
        with app.test_client() as client:
            assert client.get("/").status_code == 200


    def test_statistics_page(self, app):
        """Test the statistics page route."""
        with app.test_client() as client:
            assert client.get("/statistics").status_code == 200


    @patch("app.requests.post", autospec=True)
    def test_classify_endpoint(self, mock_post, app):
        """Test the classify route."""

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"prediction": "mocked response"}
        mock_post.return_value = mock_response

        with app.test_client() as client:
            response = client.post("/classify", json={
                "intendedNum": 1,
                "classifiedNum": 4,
                "imageData": "base64encodedstring"
            })
            assert response.status_code == 200
            assert response.json == {"prediction": "mocked response"}


    @patch("app.save_to_mongo", autospec=True)
    def test_save_results_endpoint(self, mock_save_to_mongo, app):
        """Test the save-results route."""
        mock_save_to_mongo.return_value = None

        with app.test_client() as client:
            response = client.post("/save-results", json={
                "intendedNum": 1,
                "classifiedNum": 4,
                "imageData": "base64encodedstring"
            })
            assert response.status_code == 204
            mock_save_to_mongo.assert_called_once_with({
                "intendedNum": 1,
                "classifiedNum": 4,
                "imageData": "base64encodedstring"
            })


    @patch("app.get_statistics", autospec=True)
    def test_get_stats_endpoint(self, mock_get_statistics, app):
        """Test the get-stats route."""
        mock_get_statistics.return_value = {}

        with app.test_client() as client:
            assert client.get("/get-stats").status_code == 200
