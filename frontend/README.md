# Running the new frontend

This replaces the Streamlit UI with a standalone HTML/CSS/JS frontend
talking to a small FastAPI backend that wraps the existing pipeline
(`hybrid_search`, `agent.ask_question`) — no changes to your retrieval
or agent code were needed.

1. Install the added dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Start the API (from the project root, same place `agent.py` lives):
   ```
   uvicorn backend:app --reload --port 8000
   ```
   The first question you ask will trigger the Qwen model load, same as
   before — expect that first response to be slow.

3. Serve `frontend/index.html` (don't just double-click it — some
   browsers block `fetch` from `file://`):
   ```
   cd frontend
   python -m http.server 5500
   ```
   Then open `http://localhost:5500`.

The frontend always calls the backend at `http://localhost:8000`
(set at the top of the `<script>` in `index.html`), regardless of
which port or method you use to serve the HTML itself — so step 3's
port (5500) is just for serving the page, not the API.
