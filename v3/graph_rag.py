import os
import time

# import warnings
from typing import List, TypedDict

from dotenv import load_dotenv, find_dotenv
from langchain_chroma import Chroma
from langchain_upstage import UpstageEmbeddings, ChatUpstage
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

# LangGraph
from langgraph.graph import StateGraph, END

# warnings.filterwarnings("ignore", message="Relevance scores must be between 0 and 1")

load_dotenv(find_dotenv())

PERSIST_DIR = "./chroma_insurance"

# ------------- 공용 리소스 (콜드 스타트 방지) -------------
embeddings = UpstageEmbeddings(
    api_key=os.getenv("UPSTAGE_API_KEY"),
    model="embedding-query",
)

vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory=PERSIST_DIR,
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 10},
)

model = ChatUpstage(
    model="solar-mini",
    temperature=0.1,
    # 필요 시 출력 제한:
    # max_tokens=192
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                """
                아래 규칙을 반드시 지켜라.
                - 오직 CONTEXT 내용을 기반으로 답하라.
                - CONTEXT에 해당 정보가 없거나 질문이 무관하면, 아무 설명 없이 '잘 모르겠습니다"라고만 답하라.
                - 한국어로 답하라.
                -------
                CONTEXT:\n{context}"""
            ),
        ),
        ("human", "{question}"),
    ]
)


def trim_context(docs: List[Document], limit_chars: int = 1600) -> str:
    buf, out = 0, []
    for d in docs:
        t = (d.page_content or "").strip()
        if not t:
            continue
        take_len = min(len(t), max(0, limit_chars - buf))
        if take_len > 0:
            out.append(t[:take_len])
            buf += take_len
        if buf >= limit_chars:
            break
    return "\n\n".join(out)


# ----- LangGraph 상태 -----
class RAGState(TypedDict):
    question: str
    docs: List[Document]
    context: str
    answer: str


# ----- 노드 -----
def node_retrieve(state: RAGState) -> RAGState:
    q = state["question"]
    docs = retriever.invoke(q)
    print(f"[DEBUG] retrieved docs: {len(docs)}")
    return {**state, "docs": docs}


def node_trim(state: RAGState) -> RAGState:
    docs = state.get("docs", [])
    ctx = trim_context(docs, 1600) if docs else ""
    return {**state, "context": ctx}


def node_generate(state: RAGState) -> RAGState:
    ctx = state.get("context", "").strip()
    if not ctx:
        return {**state, "answer": "해당 질문은 잘 모르겠습니다."}

    messages = prompt.format_messages(context=ctx, question=state["question"])
    out = model.invoke(messages)
    text = getattr(out, "content", None) or str(out)
    return {**state, "answer": text}


def node_fallback(state: RAGState) -> RAGState:
    # 컨텍스트가 비어있거나 의미 없는 경우 즉시 종료 응답
    return {**state, "answer": f"해당 질문은 잘 모르겠습니다."}


def has_context(state: RAGState) -> str:
    """
    context가 비어있지 않으면 node_generate로,
    비어있으면 node_fallback으로 분기
    """

    ctx = (state.get("context") or "").strip()
    return "node_generate" if ctx else "node_fallback"


# ------------ 그래프 구성 ------------
graph = StateGraph(RAGState)
graph.add_node("node_retrieve", node_retrieve)
graph.add_node("node_trim", node_trim)
graph.add_node("node_generate", node_generate)
graph.add_node("node_fallback", node_fallback)

graph.set_entry_point("node_retrieve")

graph.add_edge("node_retrieve", "node_trim")
graph.add_conditional_edges("node_trim", has_context)

graph.add_edge("node_generate", END)
graph.add_edge("node_fallback", END)

app = graph.compile()
app_until_generate = graph.compile(interrupt_before=["node_generate"])

# ------------ 실행 예시 ------------
if __name__ == "__main__":
    question = "배달-대여라이더이륜자동차 보험은 어떤 사람을 대상으로 만든 상품이야?"

    # -- 초 세기 시작
    t0 = time.perf_counter()

    result = app.invoke({"question": question, "docs": [], "context": "", "answer": ""})
    print(result["answer"])

    # -- 초 세기 끝
    print(f"\n총 소요: {time.perf_counter() - t0:.2f}s\n")
