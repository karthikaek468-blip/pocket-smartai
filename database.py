"""Simple JSON-file storage for users and recommendation history."""
import json
import threading
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from config import DATA_DIR
from models import RecommendationHistory, UserInDB

USERS_FILE = DATA_DIR / "users.json"
HISTORY_FILE = DATA_DIR / "history.json"
MAX_HISTORY_PER_USER = 50

_lock = threading.Lock()


def _load(path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save(path, data: Dict[str, Any]) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


users_db: Dict[str, UserInDB] = {
    name: UserInDB(**data) for name, data in _load(USERS_FILE).items()
}
user_recommendations: Dict[str, List[RecommendationHistory]] = {
    name: [RecommendationHistory(**item) for item in items]
    for name, items in _load(HISTORY_FILE).items()
}


def get_user(username: str) -> Optional[UserInDB]:
    return users_db.get(username.lower())


def email_taken(email: str) -> bool:
    email = email.lower()
    return any(user.email.lower() == email for user in users_db.values())


def add_user(user: UserInDB) -> None:
    with _lock:
        users_db[user.username.lower()] = user
        _save(USERS_FILE, {name: u.model_dump() for name, u in users_db.items()})


def _summarize_result(result: Dict[str, Any]) -> Dict[str, Any]:
    if "jewelry_recommendations" in result:
        categories = [item.get("item_type", "") for item in result["jewelry_recommendations"]]
    else:
        categories = [cat.get("category", "") for cat in result.get("budget_breakdown", [])]
    return {
        "total_budget": result.get("total_budget", 0),
        "total_spent": result.get("total_spent", 0),
        "remaining_budget": result.get("remaining_budget", 0),
        "categories": categories,
        "source": result.get("source", ""),
    }


def save_to_history(
    username: str, recommendation_type: str, input_data: Dict[str, Any], result: Dict[str, Any]
) -> RecommendationHistory:
    entry = RecommendationHistory(
        id=uuid.uuid4().hex,
        timestamp=datetime.now().isoformat(timespec="seconds"),
        recommendation_type=recommendation_type,
        input_summary=input_data,
        result_summary=_summarize_result(result),
        full_result=result,
    )
    with _lock:
        items = user_recommendations.setdefault(username, [])
        items.append(entry)
        del items[:-MAX_HISTORY_PER_USER]
        _save(
            HISTORY_FILE,
            {name: [h.model_dump() for h in hist] for name, hist in user_recommendations.items()},
        )
    return entry


def get_history(username: str) -> List[RecommendationHistory]:
    return sorted(user_recommendations.get(username, []), key=lambda h: h.timestamp, reverse=True)


def get_history_item(username: str, recommendation_id: str) -> Optional[RecommendationHistory]:
    for item in user_recommendations.get(username, []):
        if item.id == recommendation_id:
            return item
    return None
