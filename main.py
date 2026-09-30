"""PocketSmart AI - FastAPI backend (routes)."""
import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
from pydantic import ValidationError

from auth import (
    active_sessions, authenticate_user, create_access_token, create_session, end_session,
    get_current_active_user, get_optional_user, get_token, hash_password, remove_expired_sessions, utcnow,
)
from config import (
    ACCESS_TOKEN_EXPIRE_MINUTES, ALLOWED_IMAGE_TYPES, CORS_ORIGINS, MAX_UPLOAD_BYTES,
    STATIC_DIR, TEMPLATES_DIR, UPLOAD_DIR,
)
from database import add_user, email_taken, get_history, get_history_item, get_user, save_to_history
from gemini_utils import (
    ai_enabled, get_home_recommendations, get_jewelry_recommendations, get_party_recommendations,
)
from models import HomeBudgetInput, JewelryBudgetInput, PartyBudgetInput, RegisterUser, Token, UserInDB

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: start a background task that removes idle sessions every 5 minutes."""
    async def cleanup_expired_sessions():
        while True:
            remove_expired_sessions()
            await asyncio.sleep(300)

    task = asyncio.create_task(cleanup_expired_sessions())
    print(f"PocketSmart AI started - Gemini {'enabled' if ai_enabled() else 'NOT configured (fallback mode)'}")
    yield
    task.cancel()


app = FastAPI(title="PocketSmart: AI Budget Planner", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory=TEMPLATES_DIR)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def render(request: Request, name: str, user: Optional[UserInDB] = None, **context):
    return templates.TemplateResponse(request, name, {"user": user, "ai_enabled": ai_enabled(), **context})


def login_redirect(request: Request) -> RedirectResponse:
    return RedirectResponse(url=f"/login?next={request.url.path}", status_code=status.HTTP_302_FOUND)


def remember_in_session(username: str, key: str, value: Dict[str, Any]) -> None:
    session = active_sessions.get(username)
    if session:
        session.user_data[key] = {"timestamp": datetime.now().isoformat(timespec="seconds"), **value}


# ---------- Public pages ----------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return render(request, "index.html", get_optional_user(request))


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    if get_optional_user(request):
        return RedirectResponse(url="/dashboard", status_code=status.HTTP_302_FOUND)
    return render(request, "login.html")


@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    if get_optional_user(request):
        return RedirectResponse(url="/dashboard", status_code=status.HTTP_302_FOUND)
    return render(request, "register.html")


# ---------- Authentication ----------

@app.post("/register", status_code=status.HTTP_201_CREATED)
async def register(new_user: RegisterUser):
    username = new_user.username.lower()
    if get_user(username):
        raise HTTPException(status_code=400, detail="That username is already taken.")
    if email_taken(new_user.email):
        raise HTTPException(status_code=400, detail="An account with that email already exists.")
    add_user(UserInDB(
        username=username,
        email=new_user.email,
        full_name=new_user.full_name,
        hashed_password=hash_password(new_user.password),
    ))
    return {"message": "Account created. You can now sign in."}


@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(user.username)
    create_session(user.username, access_token)
    response = JSONResponse(content={"access_token": access_token, "token_type": "bearer"})
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        samesite="lax",
    )
    return response


@app.api_route("/logout", methods=["GET", "POST"])
async def logout(request: Request):
    end_session(get_token(request))
    response = RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    response.delete_cookie(key="access_token")
    return response


@app.get("/session-info")
async def get_session_info(current_user: UserInDB = Depends(get_current_active_user)):
    session = active_sessions.get(current_user.username)
    if session is None:
        raise HTTPException(status_code=404, detail="No active session found")
    return {
        "username": session.username,
        "login_time": session.login_time,
        "last_activity": session.last_activity,
        "session_duration": int((utcnow() - session.login_time).total_seconds() // 60),  # minutes
        "user_data": session.user_data,
    }


@app.post("/session-data")
async def update_session_data(data: Dict[str, Any], current_user: UserInDB = Depends(get_current_active_user)):
    session = active_sessions.get(current_user.username)
    if session is None:
        raise HTTPException(status_code=404, detail="No active session found")
    session.user_data.update(data)
    session.last_activity = utcnow()
    return {"message": "Session data updated", "data": session.user_data}


# ---------- Logged-in pages ----------

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    user = get_optional_user(request)
    if not user:
        return login_redirect(request)
    return render(request, "dashboard.html", user, recent=get_history(user.username)[:3])


@app.get("/home-planner", response_class=HTMLResponse)
async def home_planner(request: Request):
    user = get_optional_user(request)
    return render(request, "home_planner.html", user) if user else login_redirect(request)


@app.get("/party-planner", response_class=HTMLResponse)
async def party_planner(request: Request):
    user = get_optional_user(request)
    return render(request, "party_planner.html", user) if user else login_redirect(request)


@app.get("/jewelry-planner", response_class=HTMLResponse)
async def jewelry_planner(request: Request):
    user = get_optional_user(request)
    return render(request, "jewelry_planner.html", user) if user else login_redirect(request)


@app.get("/history", response_class=HTMLResponse)
async def history_page(request: Request):
    user = get_optional_user(request)
    return render(request, "history.html", user) if user else login_redirect(request)


# ---------- Recommendation APIs ----------

@app.post("/generate-home")
@app.post("/home-budget")
async def plan_home_budget(budget_input: HomeBudgetInput, current_user: UserInDB = Depends(get_current_active_user)):
    """Generate home interior recommendations (furniture, decor, lighting)."""
    remember_in_session(current_user.username, "last_home_budget", {
        "budget": budget_input.total_budget,
        "requirements": {
            "lights": budget_input.num_lights,
            "fans": budget_input.num_fans,
            "furniture": budget_input.num_furniture,
            "dining_tables": budget_input.num_dining_tables,
        },
    })
    result = await asyncio.to_thread(get_home_recommendations, budget_input)
    entry = save_to_history(current_user.username, "home", budget_input.model_dump(), result)
    return {**result, "history_id": entry.id}


@app.post("/generate-party")
@app.post("/party-budget")
async def plan_party_budget(budget_input: PartyBudgetInput, current_user: UserInDB = Depends(get_current_active_user)):
    """Suggest venue, catering, decoration and entertainment options for an event."""
    remember_in_session(current_user.username, "last_party_budget", {
        "budget": budget_input.total_budget,
        "party_type": budget_input.party_type,
        "guests": budget_input.num_guests,
    })
    result = await asyncio.to_thread(get_party_recommendations, budget_input)
    entry = save_to_history(current_user.username, "party", budget_input.model_dump(), result)
    return {**result, "history_id": entry.id}


async def save_upload_file(image: UploadFile) -> str:
    """Validate an uploaded outfit image and save it under static/uploads."""
    extension = ALLOWED_IMAGE_TYPES.get(image.content_type or "")
    if not extension:
        raise HTTPException(status_code=400, detail="Please upload a JPG, PNG or WEBP image.")
    data = await image.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="Image is too large (maximum 5 MB).")
    try:
        with Image.open(BytesIO(data)) as img:
            img.verify()
    except Exception:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.")
    path = UPLOAD_DIR / f"{datetime.now():%Y%m%d%H%M%S}_{uuid.uuid4().hex[:8]}.{extension}"
    path.write_bytes(data)
    return str(path)


@app.post("/generate-jewelry")
@app.post("/jewelry-budget")
async def plan_jewelry_budget(
    total_budget: float = Form(...),
    occasion: str = Form(...),
    preferences: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    current_user: UserInDB = Depends(get_current_active_user),
):
    """Suggest jewelry from text preferences and an optional outfit image."""
    try:
        budget_input = JewelryBudgetInput(
            total_budget=total_budget, occasion=occasion.strip(), preferences=(preferences or "").strip() or None
        )
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=[
            {"loc": list(err["loc"]), "msg": err["msg"]} for err in exc.errors()
        ])

    has_image = image is not None and bool(image.filename)
    image_path = await save_upload_file(image) if has_image else None

    remember_in_session(current_user.username, "last_jewelry_budget", {
        "budget": budget_input.total_budget,
        "occasion": budget_input.occasion,
        "has_image": has_image,
    })
    result = await asyncio.to_thread(get_jewelry_recommendations, budget_input, image_path)

    input_data = budget_input.model_dump()
    if image_path:
        input_data["image"] = image.filename
        result["image_url"] = f"/static/uploads/{image_path.replace(chr(92), '/').rsplit('/', 1)[-1]}"
    entry = save_to_history(current_user.username, "jewelry", input_data, result)
    return {**result, "history_id": entry.id}


@app.get("/recommendation-history")
async def get_recommendation_history(current_user: UserInDB = Depends(get_current_active_user)):
    return {"history": [
        {
            "id": item.id,
            "timestamp": item.timestamp,
            "type": item.recommendation_type,
            "input": item.input_summary,
            "summary": item.result_summary,
        }
        for item in get_history(current_user.username)
    ]}


@app.get("/recommendation-details/{recommendation_id}")
async def get_recommendation_details(
    recommendation_id: str, current_user: UserInDB = Depends(get_current_active_user)
):
    item = get_history_item(current_user.username, recommendation_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Recommendation not found")
    return {
        "id": item.id,
        "timestamp": item.timestamp,
        "type": item.recommendation_type,
        "input": item.input_summary,
        "full_result": item.full_result,
    }


if __name__ == "__main__":
    import uvicorn

    print("Starting PocketSmart: AI Budget Planner...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
