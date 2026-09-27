from langchain_text_splitters import RecursiveCharacterTextSplitter
from document_loader import source_documents


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        ""
    ]
)

chunks = text_splitter.split_documents(source_documents)

print("Number of source documents:", len(source_documents))
print("Number of chunks:", len(chunks))


for i, chunk in enumerate(chunks[:5]):
    print("\n" + "=" * 60)
    print("Chunk:", i)
    print("Content:", chunk.page_content)
    print("Metadata:", chunk.metadata)