import os
from dotenv import load_dotenv, find_dotenv
from langchain.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_upstage import UpstageEmbeddings
from langchain_chroma import Chroma

load_dotenv(find_dotenv())

FILE_PATH = "insurance_tos.pdf"
PERSIST_DIR = "./chroma_insurance"


def main():
    # 1) 문서 로드(빌드 때만)
    loader = PyMuPDFLoader(FILE_PATH)
    docs = loader.load()

    # 2) 청크 적게(짧을 수록 프롬프트 모두 빠르다고 함)
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
    splits = splitter.split_documents(docs)

    # 3) 임베딩
    embeddings = UpstageEmbeddings(
        api_key=os.getenv("UPSTAGE_API_KEY"),
        model="embedding-query",
    )

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
    )

    print("Index built & persist 완료.", PERSIST_DIR)


if __name__ == "__main__":
    main()
