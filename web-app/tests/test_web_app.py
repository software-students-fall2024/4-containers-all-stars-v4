"""Used to test web app component."""
from unittest.mock import patch, Mock
import pytest
from app import create_app
from get_statistics import get_statistics
from save_data import save_to_mongo


@pytest.fixture
def test_app():
    """Fixture for creating and configuring the Flask app."""
    return create_app()

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

    def test_home_page(self, test_app):
        """Test the home page route."""
        with test_app.test_client() as client:
            assert client.get("/").status_code == 200


    def test_statistics_page(self, test_app):
        """Test the statistics page route."""
        with test_app.test_client() as client:
            assert client.get("/statistics").status_code == 200


    @patch("app.requests.post", autospec=True)
    def test_classify_endpoint(self, mock_post, test_app):
        """Test the classify route."""

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"prediction": "mocked response"}
        mock_post.return_value = mock_response

        with test_app.test_client() as client:
            response = client.post("/classify", json={
                "intendedNum": 1,
                "classifiedNum": 4,
                "imageData": "base64encodedstring"
            })
            assert response.status_code == 200
            assert response.json == {"prediction": "mocked response"}


    @patch("app.save_to_mongo", autospec=True)
    def test_save_results_endpoint(self, mock_save_to_mongo, test_app):
        """Test the save-results route."""
        mock_save_to_mongo.return_value = None

        with test_app.test_client() as client:
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
    def test_get_stats_endpoint(self, mock_get_statistics, test_app):
        """Test the get-stats route."""
        mock_get_statistics.return_value = {}

        with test_app.test_client() as client:
            assert client.get("/get-stats").status_code == 200


    @patch("get_statistics.collection")
    def test_get_statistics_zero_documents(self, mock_collection):
        """Test get_statistics with no documents."""
        mock_collection.count_documents.return_value = 0
        result = get_statistics()
        assert "error" in result
        assert "No data available" in result["error"]

    @patch("get_statistics.collection")
    def test_get_statistics_with_data(self, mock_collection):
        """Test get_statistics with sample data."""
        def mock_count_documents(query=None):
            if not query:
                return 10
            if "$expr" in query:
                return 8
            if query == {"intended_num": 0}:
                return 5
            if query == {"intended_num": 0, "classified_num": 0}:
                return 4
            return 0
        
        mock_collection.count_documents.side_effect = mock_count_documents
        result = get_statistics()
        
        assert result["total_samples"] == 10
        assert result["correct_predictions"] == 8
        assert result["overall_accuracy"] == 80.0
        assert result["individual_digits"][0]["accuracy"] == 80.0

    @patch("save_data.collection")
    def test_save_to_mongo_invalid_data(self, mock_collection):
        """Test save_to_mongo with invalid data."""
        result = save_to_mongo({"invalid": "data"})
        assert result is False
        mock_collection.insert_one.assert_not_called()

    @patch("save_data.collection")
    def test_save_to_mongo_type_error(self, mock_collection):
        """Test save_to_mongo with data causing TypeError."""
        mock_collection.insert_one.side_effect = TypeError("Invalid type")
        data = {
            "intendedNum": "not_a_number",
            "classifiedNum": 4,
            "imageData": "base64"
        }
        result = save_to_mongo(data)
        assert result is False