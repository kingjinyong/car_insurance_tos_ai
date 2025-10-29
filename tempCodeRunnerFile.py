import os

from pathlib import Path
from datetime import datetime
from typing import Iterable, List
from dotenv import load_dotenv, find_dotenv

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_upstage import UpstageEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv(find_dotenv())

PDF_DIR = Path("./pdfs")  # 폴더 경로
PERSIST_DIR = "./chroma_insurance"


def iter_pdf_paths(root: Path) -> Iterable[Path]:
    return sorted(path for path in root.glob("*.pdf") if path.is_file())


def load_pdf(path: Path) -> List[Document]:
    loader = PyMuPDFLoader(str(path))
    docs = loader.load()
    for doc in docs:
        doc.metadata["source"] = str(path)
        doc.metadate["modified_at"] = datetime.fromtimestamp(
            path.stat().st_mtime
        ).isoformat()
    return docs


def main() -> None:
    # 1) 문서 로드(빌드 때만)
    documents: List[Document] = []
    failed: List[Path] = []

    for pdf_path in iter_pdf_paths(PDF_DIR):
        try:
            documents.extend(load_pdf(pdf_path))
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[WARN] {pdf_path.name} 로딩 실패: {exc}")
            failed.append(pdf_path)

    if failed:
        print(f"[INFO] 실패한 파일 수: {len(failed)}")

    if not documents:
        print("[INFO] 처리할 문서가 없습니다.")
        return

    # 2) 청크 적게(짧을 수록 프롬프트 모두 빠르다고 함)
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
    splits = splitter.split_documents(documents)

    # 3) 임베딩
    embeddings = UpstageEmbeddings(
        api_key=os.getenv("UPSTAGE_API_KEY"),
        model="embedding-query",
    )

    Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
    )

    print(f"[DONE] 인덱싱 완료. 총 문서 수: {len(documents)}, 청크 수: {len(splits)}")


if __name__ == "__main__":
    main()
