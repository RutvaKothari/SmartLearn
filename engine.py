import csv
import json
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_resources(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["topics"] = [x.strip() for x in row.get("topics", "").split("|") if x.strip()]
    return rows


def load_knowledge(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def clean_text(text):
    return re.sub(r"\s+", " ", str(text or "")).strip()


class SmartLearnEngine:
    # These thresholds are conservative starting values, not universal confidence probabilities.
    RESOURCE_THRESHOLD = 0.035
    ANSWER_THRESHOLD = 0.055

    def __init__(self, resources, knowledge):
        self.resources = resources
        self.knowledge = knowledge
        self.resource_texts = [
            clean_text(" ".join([
                r.get("title", ""), r.get("description", ""),
                r.get("level", ""), r.get("type", ""),
                " ".join(r.get("topics", []))
            ]))
            for r in resources
        ]
        self.knowledge_texts = [
            clean_text(d.get("title", "") + " " + d.get("content", ""))
            for d in knowledge
        ]
        # Word + character features make matching a little more tolerant of spelling variants.
        self.resource_vectorizer = TfidfVectorizer(
            lowercase=True, stop_words="english", ngram_range=(1, 2),
            sublinear_tf=True, analyzer="word"
        )
        self.resource_matrix = self.resource_vectorizer.fit_transform(self.resource_texts)
        self.knowledge_vectorizer = TfidfVectorizer(
            lowercase=True, stop_words="english", ngram_range=(1, 2),
            sublinear_tf=True, analyzer="word"
        )
        self.knowledge_matrix = self.knowledge_vectorizer.fit_transform(self.knowledge_texts)

    def recommend(self, query, level="Any level", kind="Any type", top_k=5):
        q = clean_text(query)
        if not q:
            return []
        vector = self.resource_vectorizer.transform([q])
        scores = cosine_similarity(vector, self.resource_matrix).ravel()
        ranked = scores.argsort()[::-1]
        results = []
        for idx in ranked:
            row = self.resources[idx]
            if level != "Any level" and row.get("level", "").lower() != level.lower():
                continue
            if kind != "Any type" and row.get("type", "").lower() != kind.lower():
                continue
            score = float(scores[idx])
            if score < self.RESOURCE_THRESHOLD:
                continue
            results.append({
                **row,
                "score": score,
                "topics": row.get("topics", [])
            })
            if len(results) >= top_k:
                break
        return results

    @staticmethod
    def _sentences(text):
        # Simple sentence splitting suitable for this small curated knowledge base.
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean_text(text)) if len(s.strip()) > 25]

    def answer(self, question, top_docs=3, max_sentences=3):
        q = clean_text(question)
        if not q:
            return {"supported": False, "message": "Please enter a question.", "sentences": [], "sources": [], "score": 0.0}

        vector = self.knowledge_vectorizer.transform([q])
        scores = cosine_similarity(vector, self.knowledge_matrix).ravel()
        ranked = scores.argsort()[::-1][:top_docs]
        best_score = float(scores[ranked[0]]) if len(ranked) else 0.0

        if best_score < self.ANSWER_THRESHOLD:
            return {
                "supported": False,
                "message": "I couldn't find strong supporting evidence in my knowledge base, so I won't guess.",
                "sentences": [], "sources": [], "score": best_score
            }

        candidates = []
        sources = []
        for idx in ranked:
            doc = self.knowledge[idx]
            doc_score = float(scores[idx])
            if doc_score < self.ANSWER_THRESHOLD * 0.65:
                continue
            sources.append({"title": doc["title"], "url": doc["url"]})
            for sentence in self._sentences(doc["content"]):
                sentence_vec = self.knowledge_vectorizer.transform([sentence])
                sentence_score = float(cosine_similarity(sentence_vec, vector)[0][0])
                combined = 0.65 * sentence_score + 0.35 * doc_score
                candidates.append((combined, sentence))

        candidates.sort(key=lambda x: x[0], reverse=True)
        chosen = []
        seen = set()
        for score, sentence in candidates:
            normalized = sentence.lower()
            if normalized not in seen:
                chosen.append(sentence)
                seen.add(normalized)
            if len(chosen) >= max_sentences:
                break

        if not chosen:
            return {
                "supported": False,
                "message": "A related document was found, but no sufficiently relevant sentence could be extracted. I won't guess.",
                "sentences": [], "sources": sources, "score": best_score
            }

        return {
            "supported": True, "message": "",
            "sentences": chosen, "sources": sources, "score": best_score
        }
