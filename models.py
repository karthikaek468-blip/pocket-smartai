"""Input/output schemas (Pydantic models)."""
from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, model_validator

MAX_BUDGET = 100_000_000  # ₹10 crore


class RegisterUser(BaseModel):
    username: str = Field(min_length=3, max_length=30, pattern=r"^[A-Za-z0-9_.-]+$")
    email: str = Field(min_length=5, max_length=120, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    full_name: Optional[str] = Field(default=None, max_length=80)
    password: str = Field(min_length=6, max_length=128)


class UserInDB(BaseModel):
    username: str
    email: str
    full_name: Optional[str] = None
    hashed_password: str
    disabled: bool = False


class Token(BaseModel):
    access_token: str
    token_type: str


class UserSession(BaseModel):
    username: str
    login_time: datetime
    last_activity: datetime
    token: str
    user_data: Dict[str, Any] = Field(default_factory=dict)


class HomeBudgetInput(BaseModel):
    total_budget: float = Field(gt=0, le=MAX_BUDGET)
    num_lights: int = Field(default=0, ge=0, le=200)
    num_fans: int = Field(default=0, ge=0, le=50)
    num_furniture: int = Field(default=0, ge=0, le=100)
    num_dining_tables: int = Field(default=0, ge=0, le=10)
    has_living_room: bool = False
    has_kitchen: bool = False
    has_bedroom: bool = False
    additional_requirements: Optional[str] = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def check_something_requested(self):
        counts = self.num_lights + self.num_fans + self.num_furniture + self.num_dining_tables
        rooms = self.has_living_room or self.has_kitchen or self.has_bedroom
        if counts == 0 and not rooms:
            raise ValueError("Enter at least one item quantity or select a room.")
        return self


class PartyBudgetInput(BaseModel):
    total_budget: float = Field(gt=0, le=MAX_BUDGET)
    num_guests: int = Field(ge=1, le=10000)
    party_type: str = Field(min_length=2, max_length=50)
    venue_type: Optional[str] = Field(default=None, max_length=100)
    needs_catering: bool = True
    needs_decoration: bool = True
    needs_entertainment: bool = False
    additional_requirements: Optional[str] = Field(default=None, max_length=1000)


class JewelryBudgetInput(BaseModel):
    total_budget: float = Field(gt=0, le=MAX_BUDGET)
    occasion: str = Field(min_length=2, max_length=50)
    preferences: Optional[str] = Field(default=None, max_length=500)


class RecommendationHistory(BaseModel):
    id: str
    timestamp: str
    recommendation_type: str
    input_summary: Dict[str, Any]
    result_summary: Dict[str, Any]
    full_result: Dict[str, Any]
