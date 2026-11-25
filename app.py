import streamlit as st
from engine import SmartLearnEngine, load_resources, load_knowledge

st.set_page_config(page_title="SmartLearn", page_icon="🎓", layout="wide")

@st.cache_resource
def get_engine():
    resources = load_resources("data/resources.csv")
    knowledge = load_knowledge("data/knowledge_base.json")
    return SmartLearnEngine(resources, knowledge)

engine = get_engine()

st.title("🎓 SmartLearn")
st.caption("A free, evidence-first learning resource finder. No paid API keys. No made-up AI answers.")

with st.sidebar:
    st.header("How this works")
    st.write(
        "SmartLearn searches a small, curated local knowledge base. "
        "It shows evidence and source links. If the evidence is weak, it refuses to guess."
    )
    st.info("This first version is deliberately extractive: it does not use a generative LLM.")
    st.divider()
    st.write("**Privacy:** searches run locally in this app. No query is sent to an AI API.")

tab1, tab2, tab3 = st.tabs(["🔎 Find resources", "💬 Ask from the knowledge base", "ℹ️ About accuracy"])

with tab1:
    st.subheader("Find learning resources")
    query = st.text_input(
        "What do you want to learn?",
        placeholder="e.g. beginner machine learning with Python"
    )
    c1, c2 = st.columns(2)
    with c1:
        level = st.selectbox("Your level", ["Any level", "Beginner", "Intermediate", "Advanced"])
    with c2:
        kind = st.selectbox("Resource type", ["Any type", "Documentation", "Course", "Tutorial", "Book"])

    if st.button("Find resources", type="primary", disabled=not query.strip()):
        results = engine.recommend(query, level=level, kind=kind, top_k=5)
        if not results:
            st.warning("No sufficiently relevant resources were found. Try a more specific topic.")
        else:
            for i, item in enumerate(results, 1):
                with st.container(border=True):
                    st.markdown(f"**{i}. [{item['title']}]({item['url']})**")
                    st.caption(f"{item['type']} • {item['level']} • Relevance score: {item['score']:.3f}")
                    st.write(item["description"])
                    if item.get("topics"):
                        st.write("Topics: " + ", ".join(item["topics"]))

with tab2:
    st.subheader("Ask a question")
    st.write("Answers are selected from indexed notes, not freely generated. Every answer includes its supporting source.")
    question = st.text_area(
        "Ask a question about Python, machine learning, NLP, search, RAG or deployment",
        placeholder="What is cosine similarity used for?",
        height=100
    )
    if st.button("Get evidence-based answer", type="primary", disabled=not question.strip()):
        answer = engine.answer(question)
        if not answer["supported"]:
            st.warning(answer["message"])
            st.write("Try asking about a topic covered in the knowledge base, or add a trusted document to it.")
        else:
            st.success("Relevant evidence found")
            st.markdown("**Evidence-based response**")
            for sentence in answer["sentences"]:
                st.write("• " + sentence)
            st.caption(f"Retrieval score: {answer['score']:.3f}")
            st.markdown("**Sources**")
            for source in answer["sources"]:
                st.markdown(f"- [{source['title']}]({source['url']})")
            st.caption("Read the sources before relying on the answer. The response is a short extract, not a complete explanation.")

with tab3:
    st.subheader("Accuracy and limitations")
    st.markdown("""
    **What SmartLearn does to reduce unsupported answers**
    - Searches only its bundled knowledge base.
    - Uses TF-IDF retrieval and ranks matching evidence.
    - Applies a minimum relevance threshold.
    - Refuses to answer when the evidence score is too low.
    - Displays the source title and URL for every supported response.

    **What it does not guarantee**
    - No system can promise perfect accuracy.
    - TF-IDF is a lexical retrieval method; it may miss paraphrases.
    - The bundled knowledge base is small and covers only selected beginner topics.
    - A relevant-looking passage can still be incomplete or outdated. Verify important information from the linked source.

    **How to improve it responsibly**
    1. Add high-quality, reviewed learning notes and sources.
    2. Create a test set of real student questions with expected relevant documents.
    3. Measure Precision@5 and Recall@5.
    4. Adjust the confidence threshold using validation questions, not guesses.
    """)
    st.metric("Bundled resource records", len(engine.resources))
    st.metric("Knowledge-base documents", len(engine.knowledge))
