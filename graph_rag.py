import os
import time
from typing import List, TypedDict

from dotenv import load_dotenv, find_dotenv
from langchain_chroma import Chroma
from langchain_upstage import UpstageEmbeddings, ChatUpstage
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

# LangGraph
from langgraph.graph import StateGraph, END

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

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

model = ChatUpstage(
    model="solar-mini",
    temperature=0,
    # 필요 시 출력 제한:
    # max_tokens=192
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
    return {**state, "docs": docs}


def node_trim(state: RAGState) -> RAGState:
    docs = state.get("docs", [])
    ctx = trim_context(docs, 1600) if docs else ""
    return {**state, "context": ctx}


def node_generate(state: RAGState) -> RAGState:
    ctx = state.get("context", "").strip()
    if not ctx:
        return {**state, "answer": "잘 모르겠습니다."}

    messages = prompt.format_messages(context=ctx, question=state["question"])
    out = model.invoke(messages)
    text = getattr(out, "content", None) or str(out)
    return {**state, "answer": text}


# ------------ 그래프 구성 ------------
graph = StateGraph(RAGState)
graph.add_node("node_retrieve", node_retrieve)
graph.add_node("node_trim", node_trim)
graph.add_node("node_generate", node_generate)
graph.set_entry_point("node_retrieve")
graph.add_edge("node_retrieve", "node_trim")
graph.add_edge("node_trim", "node_generate")
graph.add_edge("node_generate", END)


app = graph.compile()
app_until_generate = graph.compile(interrupt_before=["node_generate"])

# ------------ 실행 예시 ------------
if __name__ == "__main__":
    question = "가지급금을 받기 위해 필요한 절차와 서류를 단계별로 알려줘."

    print("=== 1) 일반 실행 ===")
    t0 = time.perf_counter()
    result = app.invoke({"question": question, "docs": [], "context": "", "answer": ""})
    print(result["answer"])
    print(f"\n총 소요: {time.perf_counter() - t0:.2f}s\n")

    print("=== 2) 세미 스트리밍 (generate 전까지 그래프 실행 -> LLM만 스트리밍) ===")
    # 그래프를 generate 바로 전까지 실행 (context 까지 확보)
    state = app_until_generate.invoke(
        {"question": question, "docs": [], "context": "", "answer": ""}
    )
    ctx = state.get("context", "")

    if not ctx.strip():
        print("컨텍스트 없음 -> 잘 모르겠습니다.")
    else:
        msgs = prompt.format_messages(context=ctx, question=question)
        t1 = time.perf_counter()
        first = None
        for chunk in model.stream(msgs):
            text = getattr(chunk, "content", None) or str(chunk)
            if text:
                if first is None:
                    first = time.perf_counter()
                    print(f"\n첫 토큰까지: {first - t1:.2f}s\n")
                print(text, end="", flush=True)
        print("\n")
