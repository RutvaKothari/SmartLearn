# SmartLearn — free, evidence-first learning resource finder

SmartLearn is a small, fully free starter project. It combines:
- A content-based learning-resource recommender.
- TF-IDF keyword retrieval with cosine similarity.
- A small extractive RAG-style question-answering pipeline.
- Conservative abstention when the local evidence does not match.
- Basic automated tests.
- Optional Docker packaging.

## Important accuracy design

This version does **not** call a generative LLM. It retrieves passages from `data/knowledge_base.json` and displays the most relevant sentences with source links. If retrieval is weak, it says it cannot answer rather than inventing an answer.

This reduces hallucination risk, but it does not guarantee perfect accuracy. TF-IDF is lexical search and can miss paraphrases. The knowledge base is intentionally small. Verify important details using the linked source.

## Requirements

- Python 3.10 or 3.11 recommended.
- Internet connection only to install Python packages and optionally deploy.
- No API key, paid model, paid database, or paid hosting is required for local use.

## Windows setup (VS Code / PowerShell)

1. Download and extract the ZIP.
2. Open the extracted `smartlearn_free` folder in VS Code.
3. Open Terminal → New Terminal.
4. Create a virtual environment:

   ```powershell
   py -3.11 -m venv .venv
   ```

5. Activate it:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, run this for the current terminal only:

   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   .\.venv\Scripts\Activate.ps1
   ```

6. Install dependencies:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

7. Start the app:

   ```powershell
   streamlit run app.py
   ```

8. Open the local URL printed in the terminal, usually `http://localhost:8501`.

## Run tests

From the project folder, with the virtual environment activated:

```powershell
pytest -q
```

## Project structure

```text
smartlearn_free/
├── app.py
├── engine.py
├── requirements.txt
├── Dockerfile
├── README.md
├── data/
│   ├── resources.csv
│   └── knowledge_base.json
└── tests/
    └── test_engine.py
```

## How it works

### Resource recommendation
1. Resource title, description, level, type and topics are combined.
2. TF-IDF converts resource text into vectors.
3. The user's query is converted using the same vectorizer.
4. Cosine similarity ranks resources.
5. Optional level and type filters are applied.

### Evidence-based question answering
1. The question is converted into a TF-IDF vector.
2. The knowledge-base documents are ranked by cosine similarity.
3. If the best score is below a conservative threshold, the app abstains.
4. The app selects the most relevant sentences from retrieved documents.
5. It displays the source links.

The score is a similarity score, **not** a probability of correctness.

## Add your own trusted knowledge

Edit `data/knowledge_base.json`. Each entry must contain:
- `title`: clear document title
- `url`: source URL
- `content`: your reviewed notes or permitted source text

Keep each document focused on one topic. Add the source URL. Do not paste copyrighted books or paid course content.

Edit `data/resources.csv` to add resource recommendations. Use `|` between multiple topic tags.

## Evaluation — do this before claiming accuracy

Create a set of real questions and manually label which documents are relevant. Measure Precision@5 and Recall@5 on that set. Do not tune thresholds on the final test set. Ask real students to try it and record unsupported questions and wrong retrievals.

## Free deployment

- Local use: free.
- GitHub: free for a public repository.
- Streamlit Community Cloud: offers a free deployment tier, subject to its current limits and availability.
- Docker: free to use locally.

Do not commit secrets if you later add an API. This first version has no API keys.
