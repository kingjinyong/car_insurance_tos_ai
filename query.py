import os
import time
from dotenv import load_dotenv, find_dotenv
from langchain_chroma import Chroma
from langchain_upstage import UpstageEmbeddings, ChatUpstage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


load_dotenv(find_dotenv())

PERSIST_DIR = "./chroma_insurance"

# --- 0) 전역 싱글톤처럼 미리 준비 (콜드 스타트?? 방지)

embeddings = UpstageEmbeddings(
    api_key=os.getenv("UPSTAGE_API_KEY"),
    model="embedding-query",
)

vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory=PERSIST_DIR,
)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})


# LLM: 짧고 빠르게
model = ChatUpstage(
    model="solar-pro2",  # 더 경량 모델 사용하면 더 짧은 응답속도 - 아마도..
    temperature=0,
    # 지원 시: max_tokens 또는 max_output_tokens를 128 ~ 256으로 제한
    # max_tokens=192
    # timeout=2  # 지원 시
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "다음 CONTEXT로만 한국어로 간결하게 답하세요. 모르면 '잘 모르겠습니다'라고 답하세요.\nCONTEXT:\n{context}",
        ),
        ("human", "{question}"),
    ]
)


def trim_context(docs, limit_chars=1600):
    buf, out = 0, []
    for d in docs:
        t = d.page_content.strip()
        if not t:
            continue
        take = t[: max(0, min(len(t), limit_chars - buf))]
        if take:
            out.append(take)
            buf += len(take)
        if buf >= limit_chars:
            break
    return "\n\n".join(out)


rag = (
    {
        "context": retriever | (lambda docs: trim_context(docs, 1600)),
        "question": RunnablePassthrough(),
    }
    | prompt
    | model
    | StrOutputParser()
)

if __name__ == "__main__":
    question = "가지급금을 받기 위해 필요한 절차와 서류를 단계별로 알려줘."

    print("\n=== 스트리밍 결과 ===")
    t0 = time.perf_counter()
    first_token_time = None

    for chunk in rag.stream(question):
        if first_token_time is None:
            first_token_time = time.perf_counter()
            print(f"\n⏱️ 첫 토큰까지: {first_token_time - t0:.2f}s\n")
        print(chunk, end="", flush=True)

    print("\n")
    print(f"총 소요: {time.perf_counter() - t0:.2f}s")
