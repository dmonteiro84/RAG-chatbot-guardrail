import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class FaissRetriever:
    def __init__(self, knowledge_base_path):
        """
        Initialize the FAISS retriever.
        - Load the product knowledge base from the given path.
        - Initialize the sentence transformer model for embedding text.
        """
        # Load the product knowledge base (a JSON file with product descriptions)
        self.knowledge_base = self.load_knowledge_base(knowledge_base_path)
        
        # Use a pretrained model (paraphrase-MiniLM-L6-v2) to encode text into embeddings
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        
        # Placeholder for the FAISS index, which will be built later
        self.index = None

    def load_knowledge_base(self, path):
        """
        Load the product knowledge base from a JSON file.
        Returns the loaded JSON data as a Python list of dictionaries.
        """
        with open(path, 'r') as f:
            return json.load(f)

    def get_knowledge_embeddings(self):
        """
        Generate embeddings for the product descriptions using the sentence transformer model.
        This function extracts all product descriptions from the knowledge base and returns their embeddings.
        """
        # Extract the "description" field from each product in the knowledge base
        product_descriptions = [item['description'] for item in self.knowledge_base]
        
        # Encode the descriptions into vector embeddings
        return self.model.encode(product_descriptions)

    def build_faiss_index(self):
        """
        Build the FAISS index from the embeddings of product descriptions.
        The FAISS index allows for fast similarity search based on the vector embeddings.
        """
        # Get the embeddings for the product descriptions (as float32, required by FAISS)
        knowledge_embeddings = self.get_knowledge_embeddings().astype('float32')
        
        # Get the dimension of the embeddings (the length of each vector)
        dimension = knowledge_embeddings.shape[1]

        # Create a FAISS index for similarity search using L2 (Euclidean) distance
        self.index = faiss.IndexFlatL2(dimension)
        
        # Add the embeddings to the FAISS index for fast nearest-neighbor search
        self.index.add(knowledge_embeddings)

    def search(self, query, k=3):
        """
        Perform a search for the top k similar products based on the user query.
        - The query is encoded into an embedding, and the FAISS index is searched for the most similar products.
        - Returns the top k products from the knowledge base.
        """
        # Convert the user query into a vector embedding using the sentence transformer model
        query_embedding = self.model.encode([query]).astype('float32')

        # Ensure the FAISS index is built before searching (build it if it doesn't exist)
        if self.index is None:
            self.build_faiss_index()

        # Perform a search in the FAISS index, retrieving the top k nearest neighbors
        D, I = self.index.search(query_embedding, k)  # D = distances, I = indices of nearest neighbors
        
        # Get the top k products from the knowledge base using the indices returned by FAISS
        results = [self.knowledge_base[i] for i in I[0]]
        return results
    
    def retrieve_fact(self, query):
        """
        Retrieve the most relevant fact based on the user query.
        This function performs a search on the FAISS index and returns the top relevant fact.
        """
        results = self.search(query)  # Perform search using the FAISS index
        if results:
            return results[0]  # Return the top fact from the search results
        else:
            return "No relevant facts found."  # Fallback if no results are found

# Example usage:
if __name__ == "__main__":
    # Initialize the retriever with a knowledge base (products.json)
    retriever = FaissRetriever('data/products.json')
    
    # Example user query
    query = "Tell me about the credit card."
    
    # Perform a search and retrieve the top 3 results
    results = retriever.search(query)
    
    # Print the search results
    print(results)
