from langchain.vectorstores import Chroma
from Embeddings import embeddings_function
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders.pdf import PyPDFDirectoryLoader

CHROMA_PATH = "chroma"
DATA_PATH="data"

def main():
    # Create (or update) the data store.
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    document= document_loader.load()
    update_doc(document)


def update_doc(document):
    """
    Splits a document into smaller chunks based on a specified size.
    """
    # Initialize the text splitter with a chunk size of 1000 characters
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    # Example document to split

    # Split the document
    if not document:
        print("❌ No documents to process")
        return
    
    chunks = text_splitter.split_documents(document)


    db= Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings_function()
    )
    
    new_chunks_with_ids=calculate_chunk_ids(chunks)
    

    existing_chunks=db.get(include=[])
    existing_ids=set(existing_chunks['ids'])
    print(f"length of existing db docs: {len(existing_ids)}")

    new_chunks=[]
    for i in new_chunks_with_ids:
        if i.metadata['id'] not in existing_ids:
            new_chunks.append(i)

    if len(new_chunks):
        print(f"Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("No new documents to add")

def calculate_chunk_ids(chunks):

    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata["source"]
        page = chunk.metadata["page"]
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks

if __name__ == "__main__":
    main()