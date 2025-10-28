import os
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_upstage import UpstageDocumentParseLoader, UpstageDocumentParseParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_upstage import UpstageEmbeddings, ChatUpstage
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.document_loaders import PyMuPDFLoader
from operator import itemgetter
import langchain  # 👈 1. langchain 라이브러리 추가


langchain.debug = True  # 👈
# Upstage API 키 설정 (환경 변수에 설정했다고 가정)
# os.environ["UPSTAGE_API_KEY"] = "YOUR_API_KEY"

# 파일 경로
FILE_PATH = "insurance_tos.pdf"  # 실제 파일 경로로 변경하세요.
load_dotenv(find_dotenv())


# 1. 파일 불러오기 (Load) 및 파싱 (Parse)
# UpstageLayoutAnalysisLoader를 사용하면 문서 구조 정보(레이아웃)를 유지할 수 있습니다.
loader = PyMuPDFLoader("insurance_tos.pdf")
docs = loader.load()

# 2. 청크 분할 (Split)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

print(f"원본 문서 개수: {len(docs)}")
print(f"분할된 청크 개수: {len(splits)}")


embeddings = UpstageEmbeddings(
    api_key=os.getenv("UPSTAGE_API_KEY"), model="embedding-query"
)


vectorstore = Chroma.from_documents(splits, embeddings)
print("벡터스토어에 문서 저장 완료")

# docs = vectorstore.similarity_search("가지급금이 무엇인지 설명해주세요.")


retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # 상위 3개 문서 검색

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
        너는 인공지능 챗봇으로, 주어진 문서를 정확하게 이해해서 답변해야 해.
        문서에 있는 내용으로만 답변하고, 내용이 없으면 '잘 모르겠습니다'라고 답해.
        답변은 꼭 한글로 해줘.
        ---
        CONTEXT:
        {context}
        """,
        ),
        ("human", "{question}"),
    ]
)

model = ChatUpstage(model="solar-pro2")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {
        "context": retriever | format_docs,  # 검색된 문서 컨텍스트 생성
        "question": RunnablePassthrough(),  # 사용자의 질문
    }
    | prompt
    | model
    | StrOutputParser()
)

response = rag_chain.invoke(
    "가지급금을 받기 위해서는 어떤 절차가 주어지고 어떤 서류가 필요한가요? 모든 것을 설명해주세요."
)

print(response)
# llm = ChatUpstage(api_key=os.getenv("UPSTAGE_API_KEY"))
