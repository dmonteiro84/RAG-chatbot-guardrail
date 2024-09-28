import unittest
from src.retrieval import FaissRetriever

class TestFaissRetriever(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Set up the FAISS retriever with the knowledge base before running tests.
        """
        cls.retriever = FaissRetriever("data/products.json")  # Assuming the JSON file is the knowledge base

    def test_retriever_basic_query(self):
        """
        Test if the retriever returns relevant results for a basic query.
        """
        query = "Tell me about credit cards"
        results = self.retriever.search(query, k=3)  # Retrieve 3 results

        self.assertIsInstance(results, list, "The results should be returned as a list.")
        self.assertGreaterEqual(len(results), 1, "The retriever should return at least 1 result.")
        self.assertIn("credit card", results[0]["product_name"].lower(), "The top result should be about credit cards.")

    def test_retriever_no_results(self):
        """
        Test if the retriever handles queries that have no results.
        """
        query = "Non-existent product"
        results = self.retriever.search(query, k=3)

        self.assertIsInstance(results, list, "The results should be returned as a list.")
        self.assertEqual(len(results), 0, "The retriever should return an empty list if no products are found.")

    def test_retriever_complex_query(self):
        """
        Test if the retriever can handle more complex queries and return relevant results.
        """
        query = "What are the best home loans for low interest rates?"
        results = self.retriever.search(query, k=3)

        self.assertIsInstance(results, list, "The results should be returned as a list.")
        self.assertGreaterEqual(len(results), 1, "The retriever should return at least 1 result.")
        self.assertIn("home loan", results[0]["product_name"].lower(), "The top result should be about home loans.")

    def test_result_format(self):
        """
        Test if the retrieval results are returned in the expected format (i.e., they contain required fields).
        """
        query = "Tell me about personal loans"
        results = self.retriever.search(query, k=1)

        self.assertIsInstance(results, list, "The results should be returned as a list.")
        self.assertGreaterEqual(len(results), 1, "The retriever should return at least 1 result.")
        self.assertIn("product_name", results[0], "Each result should contain the 'product_name' field.")
        self.assertIn("description", results[0], "Each result should contain the 'description' field.")

if __name__ == "__main__":
    unittest.main()
