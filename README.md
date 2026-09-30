# PocketSmart AI: Your Smart Budget & Recommendation Assistant

A FastAPI + Gemini web app that gives budget-based recommendations for home interiors,
party planning and jewelry (with optional outfit image), with shopping links to Amazon,
Flipkart, IKEA, Swiggy, Zomato, OYO, BlueStone, Tanishq and more.

## Setup

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env      # then put your Gemini API key in .env
uvicorn main:app --reload
```

Open http://127.0.0.1:8000, create an account, and use the planners.

Without a `GOOGLE_API_KEY` the app still runs: it uses the built-in fallback planner
(simulated catalog with typical Indian prices) and labels the results as such.

## Project structure

| File | Purpose |
|---|---|
| `main.py` | FastAPI app and routes (pages, auth, planners, history, sessions) |
| `gemini_utils.py` | Gemini prompts, JSON parsing, budget checks, shopping links, fallback recommendations |
| `models.py` | Pydantic input/output schemas |
| `auth.py` | Password hashing, JWT tokens, session management |
| `database.py` | JSON-file storage for users and history (`data/`) |
| `config.py` | Settings loaded from `.env` |
| `templates/` | Jinja2 pages: index, login, register, dashboard, planners, history |
| `static/` | `style.css`, `app.js` (renders results), `uploads/` (outfit images) |

## Routes

| Route | Description |
|---|---|
| `GET /` | Landing page |
| `GET/POST /register`, `GET /login`, `POST /token`, `GET/POST /logout` | Authentication |
| `GET /dashboard` | Dashboard with recent activity |
| `GET /home-planner`, `/party-planner`, `/jewelry-planner` | Planner pages |
| `POST /generate-home` (alias `/home-budget`) | Home interior recommendations (JSON) |
| `POST /generate-party` (alias `/party-budget`) | Party budget plan (JSON) |
| `POST /generate-jewelry` (alias `/jewelry-budget`) | Jewelry recommendations (multipart form + optional image) |
| `GET /history`, `/recommendation-history`, `/recommendation-details/{id}` | Recommendation history |
| `GET /session-info`, `POST /session-data` | Session data |
| `GET /docs` | Interactive API docs |
