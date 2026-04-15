import chromadb
import os

try:
    persist_directory = os.path.join('data', 'test_vector_store')
    os.makedirs(persist_directory, exist_ok=True)
    print(f"Attempting to create PersistentClient at {persist_directory}")
    client = chromadb.PersistentClient(path=persist_directory)
    print("Client created successfully!")
    collection = client.get_or_create_collection(name="test_collection")
    print("Collection created successfully!")
    collection.add(ids=["test1"], documents=["test document"])
    print("Document added successfully!")
    results = collection.query(query_texts=["test"], n_results=1)
    print(f"Query results: {results}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
