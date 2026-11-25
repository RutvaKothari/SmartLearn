from engine import SmartLearnEngine, load_resources, load_knowledge

def build_engine():
    return SmartLearnEngine(
        load_resources("data/resources.csv"),
        load_knowledge("data/knowledge_base.json")
    )

def test_known_question_returns_evidence():
    result = build_engine().answer("What does cosine similarity measure?")
    assert result["supported"] is True
    assert result["sentences"]
    assert result["sources"]

def test_unrelated_question_abstains():
    result = build_engine().answer("Who won the cricket world cup in 1983?")
    assert result["supported"] is False

def test_recommendation_returns_python_resource():
    result = build_engine().recommend("beginner Python programming")
    assert result
    assert "Python" in result[0]["title"]
