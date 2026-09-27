from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader


DATA_PATH = Path("data/documents")


source_documents = []

for pdf_file in DATA_PATH.glob("*.pdf"):

    loader = PyPDFLoader(str(pdf_file))

    documents = loader.load()

    for doc in documents:

        doc.metadata["document_id"] = pdf_file.stem
        doc.metadata["title"] = pdf_file.stem
        doc.metadata["source_file"] = pdf_file.name

        source_documents.append(doc)