"""Recommendation engine: Gemini prompts, JSON parsing, shopping links and fallbacks.

When Gemini is not configured or fails, a rule-based planner (simulated product
catalog with typical Indian prices) produces recommendations so the app keeps working.
"""
import json
import logging
import math
import re
import urllib.parse
from collections import Counter
from typing import Any, Callable, Dict, List, Optional

from PIL import Image

from config import GEMINI_API_KEY, GEMINI_MODEL
from models import HomeBudgetInput, JewelryBudgetInput, PartyBudgetInput

logger = logging.getLogger("pocketsmart")

_client = None
_types = None
if GEMINI_API_KEY:
    try:
        from google import genai
        from google.genai import types as _types

        _client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as exc:  # missing package or bad configuration
        logger.warning("Gemini could not be initialised (%s). Using fallback recommendations.", exc)
else:
    logger.warning("GOOGLE_API_KEY is not set. Using fallback recommendations.")


def ai_enabled() -> bool:
    return _client is not None


def _generate(contents: List[Any]) -> str:
    response = _client.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=_types.GenerateContentConfig(response_mime_type="application/json", temperature=0.4),
    )
    return response.text or ""


def extract_json_from_response(text: str) -> Dict[str, Any]:
    """Parse JSON from a model response, tolerating ```json fences or extra text."""
    cleaned = re.sub(r"```(?:json)?", "", text or "").strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("No JSON object found in AI response")
        data = json.loads(cleaned[start:end + 1])
    if not isinstance(data, dict):
        raise ValueError("AI response JSON is not an object")
    return data


# ---------- Shopping links ----------

PLATFORMS = {
    "amazon": ("Amazon", "https://www.amazon.in/s?k={q}"),
    "flipkart": ("Flipkart", "https://www.flipkart.com/search?q={q}"),
    "ikea": ("IKEA", "https://www.ikea.com/in/en/search/?q={q}"),
    "myntra": ("Myntra", "https://www.myntra.com/search?q={q}"),
    "ajio": ("Ajio", "https://www.ajio.com/search/?text={q}"),
    "meesho": ("Meesho", "https://www.meesho.com/search?q={q}"),
    "bigbasket": ("BigBasket", "https://www.bigbasket.com/ps/?q={q}"),
    "swiggy": ("Swiggy", "https://www.swiggy.com/search?query={q}"),
    "zomato": ("Zomato", "https://www.zomato.com/search?q={q}"),
    "bookmyshow": ("BookMyShow", "https://in.bookmyshow.com/search?q={q}"),
    "google": ("Google", "https://www.google.com/search?q={q}"),
    "booking": ("Booking.com", "https://www.booking.com/search.html?ss={q}"),
    "makemytrip": ("MakeMyTrip", "https://www.makemytrip.com/hotels/hotel-listing/?searchText={q}"),
    "oyorooms": ("OYO", "https://www.oyorooms.com/search/?location={q}"),
    "nobroker": ("NoBroker", "https://www.nobroker.in/property/search?searchTerm={q}"),
    "bluestone": ("BlueStone", "https://www.bluestone.com/search.html?query={q}"),
    "tanishq": ("Tanishq", "https://www.tanishq.co.in/search?q={q}"),
    "caratlane": ("CaratLane", "https://www.caratlane.com/search?q={q}"),
    "melorra": ("Melorra", "https://www.melorra.com/search?q={q}"),
}

HOME_PLATFORMS = ["amazon", "flipkart", "ikea"]
JEWELRY_PLATFORMS = ["amazon", "flipkart", "bluestone", "tanishq", "caratlane", "melorra", "meesho"]
VENUE_PLATFORMS = ["google", "booking", "makemytrip", "oyorooms", "nobroker"]
PARTY_CATEGORY_PLATFORMS = {
    "venue": VENUE_PLATFORMS,
    "catering": ["swiggy", "zomato"],
    "food": ["swiggy", "zomato", "bigbasket", "amazon", "flipkart"],
    "drinks": ["swiggy", "zomato", "bigbasket", "amazon", "flipkart"],
    "decoration": ["amazon", "flipkart", "meesho", "myntra"],
    "entertainment": ["bookmyshow", "amazon", "flipkart"],
    "gifts": ["amazon", "flipkart", "myntra", "meesho"],
    "return_gifts": ["amazon", "flipkart", "myntra", "meesho"],
    "photography": ["google", "amazon", "flipkart"],
    "music": ["amazon", "flipkart", "bookmyshow"],
    "games": ["amazon", "flipkart"],
    "accessories": ["amazon", "flipkart", "myntra", "meesho"],
    "transportation": ["makemytrip", "google"],
    "contingency": ["amazon", "flipkart", "google"],
}
DEFAULT_PLATFORMS = ["amazon", "flipkart", "google"]


def build_links(search_terms: str, platforms: List[str]) -> Dict[str, Dict[str, str]]:
    if not search_terms:
        return {}
    query = urllib.parse.quote_plus(search_terms)
    return {
        key: {"label": PLATFORMS[key][0], "url": PLATFORMS[key][1].format(q=query)}
        for key in platforms if key in PLATFORMS
    }


def _party_platforms(category: str) -> List[str]:
    key = category.lower().strip().replace(" ", "_")
    return PARTY_CATEGORY_PLATFORMS.get(key, DEFAULT_PLATFORMS)


# ---------- Result normalisation ----------

def _num(value: Any) -> float:
    try:
        number = float(str(value).replace(",", "").replace("₹", "").strip())
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) and number > 0 else 0.0


def _int(value: Any, default: int = 1) -> int:
    number = _num(value)
    return int(number) if number >= 1 else default


def _text(value: Any, limit: int = 300) -> str:
    return str(value or "").strip()[:limit]


def _text_list(values: Any, limit: int = 8) -> List[str]:
    if not isinstance(values, list):
        return []
    return [_text(v) for v in values if _text(v)][:limit]


def _finish_totals(result: Dict[str, Any], budget: float, spent: float) -> Dict[str, Any]:
    result["total_budget"] = round(budget, 2)
    result["total_spent"] = round(spent, 2)
    result["remaining_budget"] = round(budget - spent, 2)
    result["within_budget"] = spent <= budget + 0.01
    if not result["within_budget"]:
        result.setdefault("additional_suggestions", []).insert(
            0, f"This plan is ₹{spent - budget:,.0f} over budget. Reduce quantities or pick cheaper items."
        )
    return result


def finalize_breakdown(raw: Dict[str, Any], budget: float, platforms: Callable[[str], List[str]]) -> Dict[str, Any]:
    """Validate a budget_breakdown result, recompute totals and add shopping links."""
    breakdown, table, spent = [], [], 0.0
    for category in raw.get("budget_breakdown") or []:
        if not isinstance(category, dict):
            continue
        name = _text(category.get("category"), 60) or "Misc"
        items, category_total = [], 0.0
        for item in category.get("items") or []:
            if not isinstance(item, dict):
                continue
            price = _num(item.get("estimated_price"))
            quantity = _int(item.get("quantity"))
            terms = _text(item.get("search_terms") or item.get("name"), 120)
            line_total = round(price * quantity, 2)
            items.append({
                "name": _text(item.get("name"), 120) or "Item",
                "description": _text(item.get("description")),
                "estimated_price": round(price, 2),
                "quantity": quantity,
                "line_total": line_total,
                "search_terms": terms,
                "shopping_links": build_links(terms, platforms(name)),
            })
            category_total += line_total
        if not items:
            continue
        breakdown.append({
            "category": name,
            "allocation": round(_num(category.get("allocation")) or category_total, 2),
            "category_total": round(category_total, 2),
            "items": items,
        })
        table.append({
            "category": name,
            "items_count": len(items),
            "total_cost": round(category_total, 2),
            "percentage_of_budget": round(category_total / budget * 100, 1) if budget else 0,
        })
        spent += category_total

    result = {
        "budget_breakdown": breakdown,
        "calculation_table": table,
        "additional_suggestions": _text_list(raw.get("additional_suggestions")),
    }
    if "venue_suggestions" in raw:
        venues = []
        for venue in raw.get("venue_suggestions") or []:
            if not isinstance(venue, dict):
                continue
            terms = _text(venue.get("search_terms") or venue.get("name"), 120)
            venues.append({
                "name": _text(venue.get("name"), 120) or "Venue",
                "type": _text(venue.get("type"), 60),
                "capacity": _int(venue.get("capacity"), 0),
                "estimated_cost": round(_num(venue.get("estimated_cost")), 2),
                "search_links": build_links(terms, VENUE_PLATFORMS),
            })
        result["venue_suggestions"] = venues[:5]
    return _finish_totals(result, budget, spent)


def finalize_jewelry(raw: Dict[str, Any], budget: float) -> Dict[str, Any]:
    items, spent = [], 0.0
    for item in raw.get("jewelry_recommendations") or []:
        if not isinstance(item, dict):
            continue
        price = _num(item.get("estimated_price"))
        terms = _text(item.get("search_terms") or item.get("item_type"), 120)
        items.append({
            "item_type": _text(item.get("item_type"), 80) or "Jewelry",
            "description": _text(item.get("description")),
            "style": _text(item.get("style"), 60),
            "estimated_price": round(price, 2),
            "search_terms": terms,
            "shopping_links": build_links(terms, JEWELRY_PLATFORMS),
        })
        spent += price
    result = {
        "jewelry_recommendations": items,
        "styling_tips": _text_list(raw.get("styling_tips")),
        "additional_suggestions": _text_list(raw.get("additional_suggestions")),
    }
    analysis = raw.get("outfit_analysis")
    if isinstance(analysis, dict):
        result["outfit_analysis"] = {
            "colors": _text_list(analysis.get("colors"), 6),
            "style": _text(analysis.get("style"), 60),
            "formality": _text(analysis.get("formality"), 60),
        }
    return _finish_totals(result, budget, spent)


def _has_results(result: Dict[str, Any]) -> bool:
    return bool(result.get("budget_breakdown") or result.get("jewelry_recommendations"))


def _run(contents: Callable[[], List[Any]], finalize: Callable[[Dict[str, Any]], Dict[str, Any]],
         fallback: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
    """Try Gemini first; use the rule-based fallback when it is unavailable or returns too little."""
    note = "Gemini API key not configured - showing sample recommendations from the built-in catalog."
    if _client is not None:
        try:
            result = finalize(extract_json_from_response(_generate(contents())))
            if _has_results(result):
                result["source"] = "gemini"
                result["model"] = GEMINI_MODEL
                return result
            note = "Gemini returned too few results - showing built-in recommendations instead."
        except Exception as exc:
            logger.warning("Gemini request failed: %s", exc)
            note = "Gemini is unavailable right now - showing built-in recommendations instead."
    result = finalize(fallback())
    result["source"] = "fallback"
    result["source_note"] = note
    return result


def _floor10(value: float) -> float:
    return max(0.0, math.floor(value / 10) * 10)


# ---------- Home planner ----------

HOME_JSON_FORMAT = """{
  "budget_breakdown": [
    {
      "category": "Lighting",
      "allocation": 0,
      "items": [
        {"name": "", "description": "", "estimated_price": 0, "quantity": 0, "search_terms": ""}
      ]
    }
  ],
  "additional_suggestions": [""]
}"""


def _home_prompt(inp: HomeBudgetInput) -> str:
    rooms = [name for flag, name in [
        (inp.has_living_room, "Living room"), (inp.has_kitchen, "Kitchen"), (inp.has_bedroom, "Bedroom"),
    ] if flag]
    return f"""
I need interior design product recommendations for a home in India with a total budget of ₹{inp.total_budget:,.2f}.

Requirements:
- {inp.num_lights} lights/lighting fixtures
- {inp.num_fans} ceiling fans
- {inp.num_furniture} furniture pieces
- {inp.num_dining_tables} dining tables

Rooms to consider: {", ".join(rooms) or "None specified"}
Additional requirements: {inp.additional_requirements or "None"}

Give a budget breakdown with one category per requirement (skip requirements with quantity 0)
and one category per selected room for decor essentials. Balance functionality, style and price.
Recommend products available in India from Indian brands or platforms like IKEA, Amazon India and Flipkart.
"estimated_price" is the price in INR for ONE unit; "quantity" is the number of units.
The sum of estimated_price x quantity across all items MUST NOT exceed ₹{inp.total_budget:,.2f}.
"search_terms" must be a short product search query for Indian shopping sites.

Respond only with JSON in this structure:
{HOME_JSON_FORMAT}
"""


# (name, description, unit price in ₹, search terms), cheapest first
HOME_CATALOG = {
    "Lighting": ("num_lights", [
        ("LED Batten Tube Light 20W", "Energy-efficient tube light for everyday lighting.", 350, "LED batten tube light 20W"),
        ("LED Panel Ceiling Light 15W", "Slim ceiling panel light with a clean modern look.", 900, "LED panel ceiling light 15W"),
        ("Decorative Pendant Light", "Stylish pendant lamp for living or dining areas.", 2500, "decorative pendant ceiling light"),
    ]),
    "Ceiling Fans": ("num_fans", [
        ("Standard 1200mm Ceiling Fan", "Reliable high-speed fan for any room.", 1800, "1200mm ceiling fan"),
        ("Energy-Saving BLDC Ceiling Fan", "5-star rated BLDC fan that cuts electricity bills.", 3500, "BLDC ceiling fan 5 star"),
        ("Designer Ceiling Fan with Remote", "Premium decorative fan with remote control.", 6500, "designer ceiling fan with remote"),
    ]),
    "Furniture": ("num_furniture", [
        ("Folding Chair", "Compact, sturdy chair that folds away when not in use.", 1200, "folding chair"),
        ("Engineered Wood Study / Side Table", "Versatile table for work or as a side unit.", 4500, "engineered wood study table"),
        ("Fabric 3-Seater Sofa", "Comfortable sofa for the living room.", 18000, "3 seater fabric sofa"),
    ]),
    "Dining Tables": ("num_dining_tables", [
        ("4-Seater Engineered Wood Dining Set", "Space-saving dining table with 4 chairs.", 8000, "4 seater dining table set"),
        ("6-Seater Sheesham Wood Dining Set", "Solid wood dining set for families.", 18000, "6 seater sheesham dining table"),
        ("6-Seater Marble Top Dining Set", "Premium marble-top dining set.", 35000, "6 seater marble top dining table"),
    ]),
    "Living Room Decor": ("has_living_room", [
        ("Curtain Set (2 panels)", "Polyester curtains to finish the room.", 800, "curtains set of 2"),
        ("Blackout Curtains + Area Rug", "Blackout curtains with a cotton rug.", 3500, "blackout curtains and area rug"),
        ("Curtains, Rug & Wall Art Combo", "Complete decor refresh for the living room.", 8000, "living room decor combo wall art rug"),
    ]),
    "Kitchen": ("has_kitchen", [
        ("Spice & Utility Organiser", "Keeps the counter tidy and organised.", 800, "kitchen spice organiser rack"),
        ("Stainless Steel Storage Rack", "Multi-tier rack for utensils and appliances.", 2500, "stainless steel kitchen storage rack"),
        ("Modular Kitchen Trolley Set", "Pull-out storage for a modular kitchen.", 6000, "modular kitchen trolley basket set"),
    ]),
    "Bedroom": ("has_bedroom", [
        ("Cotton Double Bedsheet Set", "Soft bedsheet with 2 pillow covers.", 1100, "cotton double bedsheet"),
        ("Bedsheet Set + Bedside Lamp", "Bedding and a warm reading lamp.", 2500, "bedsheet and bedside lamp"),
        ("Bedside Table, Lamp & Bedding", "Complete bedroom comfort set.", 7000, "bedside table lamp bedding set"),
    ]),
}


def _home_fallback(inp: HomeBudgetInput) -> Dict[str, Any]:
    needs = []
    for category, (field, options) in HOME_CATALOG.items():
        value = getattr(inp, field)
        quantity = 1 if isinstance(value, bool) else value
        if quantity:
            needs.append((category, options, quantity))

    budget = inp.total_budget
    mid_total = sum(options[1][2] * qty for _, options, qty in needs)
    minimum = sum(options[0][2] * qty for _, options, qty in needs)
    breakdown = []
    for category, options, qty in needs:
        allocation = budget * 0.95 * (options[1][2] * qty / mid_total)
        affordable = [o for o in options if o[2] * qty <= allocation]
        name, description, price, terms = affordable[-1] if affordable else options[0]
        breakdown.append({
            "category": category,
            "allocation": round(allocation, 2),
            "items": [{"name": name, "description": description, "estimated_price": price,
                       "quantity": qty, "search_terms": terms}],
        })

    suggestions = [
        "Compare prices across Amazon, Flipkart and IKEA before buying - sale prices vary a lot.",
        "Look for festive sales (Diwali, Big Billion Days, Prime Day) for big-ticket furniture.",
        "Prioritise essential items first and add decor pieces later.",
    ]
    if budget < minimum:
        suggestions.insert(0, f"The cheapest options for these quantities need about ₹{minimum:,.0f}. "
                              "Consider reducing quantities or buying second-hand furniture.")
    return {"budget_breakdown": breakdown, "additional_suggestions": suggestions}


def get_home_recommendations(inp: HomeBudgetInput) -> Dict[str, Any]:
    return _run(
        lambda: [_home_prompt(inp)],
        lambda raw: finalize_breakdown(raw, inp.total_budget, lambda _: HOME_PLATFORMS),
        lambda: _home_fallback(inp),
    )


# ---------- Party planner ----------

PARTY_JSON_FORMAT = """{
  "budget_breakdown": [
    {
      "category": "venue",
      "allocation": 0,
      "items": [
        {"name": "", "description": "", "estimated_price": 0, "quantity": 0, "search_terms": ""}
      ]
    }
  ],
  "venue_suggestions": [
    {"name": "", "type": "", "capacity": 0, "estimated_cost": 0, "search_terms": ""}
  ],
  "additional_suggestions": [""]
}"""


def _party_prompt(inp: PartyBudgetInput) -> str:
    def yes(flag: bool) -> str:
        return "Yes" if flag else "No"

    return f"""
I need party planning recommendations for India with a total budget of ₹{inp.total_budget:,.2f}.

Party details:
- Type: {inp.party_type}
- Number of guests: {inp.num_guests}
- Venue type: {inp.venue_type or "Not specified"}
- Catering needed: {yes(inp.needs_catering)}
- Decoration needed: {yes(inp.needs_decoration)}
- Entertainment needed: {yes(inp.needs_entertainment)}

Additional requirements: {inp.additional_requirements or "None"}

Allocate the budget proportionally across venue, catering, decoration and entertainment
(only the needed ones), plus a small contingency. Tailor suggestions to the event type.
Use Indian services and typical Indian costs: Swiggy/Zomato for food, OYO/MakeMyTrip for venues
and stays, BookMyShow for entertainment, Amazon/Flipkart for decorations.
Use lowercase category names such as venue, catering, decoration, entertainment, contingency.
"estimated_price" is the INR price for ONE unit (e.g. per plate); "quantity" is the number of units.
The sum of estimated_price x quantity MUST NOT exceed ₹{inp.total_budget:,.2f}.

Respond only with JSON in this structure:
{PARTY_JSON_FORMAT}
"""


PARTY_SPLITS = {
    "wedding": {"venue": 30, "catering": 40, "decoration": 18, "entertainment": 7, "contingency": 5},
    "corporate": {"venue": 35, "catering": 35, "decoration": 8, "entertainment": 12, "contingency": 10},
    "birthday": {"venue": 20, "catering": 40, "decoration": 15, "entertainment": 15, "contingency": 10},
    "default": {"venue": 25, "catering": 40, "decoration": 15, "entertainment": 10, "contingency": 10},
}
PARTY_KEYWORDS = {
    "wedding": ["wedding", "marriage", "engagement", "reception", "sangeet", "mehendi"],
    "corporate": ["corporate", "office", "conference", "team", "business", "launch"],
    "birthday": ["birthday", "kids", "bday"],
}
HOME_VENUE_WORDS = ["home", "house", "terrace", "own", "apartment", "flat"]


def _party_fallback(inp: PartyBudgetInput) -> Dict[str, Any]:
    party = inp.party_type.lower()
    kind = next((k for k, words in PARTY_KEYWORDS.items() if any(w in party for w in words)), "default")
    split = dict(PARTY_SPLITS[kind])
    venue_text = (inp.venue_type or "").lower()
    at_home = any(w in venue_text for w in HOME_VENUE_WORDS)
    for needed, key in [(not at_home, "venue"), (inp.needs_catering, "catering"),
                        (inp.needs_decoration, "decoration"), (inp.needs_entertainment, "entertainment")]:
        if not needed:
            split.pop(key)
    total_share = sum(split.values())
    alloc = {k: inp.total_budget * v / total_share for k, v in split.items()}
    guests, label = inp.num_guests, inp.party_type.strip().title()

    breakdown = []
    if "venue" in alloc:
        hall = "Restaurant private dining area" if guests <= 30 else (
            "Banquet / party hall" if guests <= 150 else "Marriage hall / convention centre")
        breakdown.append({"category": "venue", "allocation": alloc["venue"], "items": [{
            "name": hall, "description": f"Venue booking for {guests} guests.",
            "estimated_price": _floor10(alloc["venue"]), "quantity": 1,
            "search_terms": f"{hall.split(' /')[0]} for {guests} guests"}]})
    if "catering" in alloc:
        per_plate = max(1, math.floor(alloc["catering"] / guests))
        style = "Home-style meal / bulk food order" if per_plate < 250 else (
            "Buffet catering (veg + non-veg)" if per_plate < 800 else "Premium multi-cuisine buffet")
        breakdown.append({"category": "catering", "allocation": alloc["catering"], "items": [{
            "name": style, "description": f"About ₹{per_plate:,} per plate for {guests} guests.",
            "estimated_price": per_plate, "quantity": guests,
            "search_terms": f"party catering for {guests} people"}]})
    if "decoration" in alloc:
        breakdown.append({"category": "decoration", "allocation": alloc["decoration"], "items": [
            {"name": f"{label} theme decoration package", "description": "Backdrop, lighting and stage/table decor.",
             "estimated_price": _floor10(alloc["decoration"] * 0.6), "quantity": 1,
             "search_terms": f"{label} decoration items"},
            {"name": "Balloons, banners & table decor kit", "description": "DIY decor kit to finish the space.",
             "estimated_price": _floor10(alloc["decoration"] * 0.4), "quantity": 1,
             "search_terms": f"{label} balloon decoration kit"},
        ]})
    if "entertainment" in alloc:
        breakdown.append({"category": "entertainment", "allocation": alloc["entertainment"], "items": [
            {"name": "DJ / music system rental", "description": "Sound system with a DJ or curated playlist.",
             "estimated_price": _floor10(alloc["entertainment"] * 0.7), "quantity": 1,
             "search_terms": "party speaker music system"},
            {"name": "Party games & photo booth props", "description": "Games and props to keep guests engaged.",
             "estimated_price": _floor10(alloc["entertainment"] * 0.3), "quantity": 1,
             "search_terms": "party games photo booth props"},
        ]})
    breakdown.append({"category": "contingency", "allocation": alloc["contingency"], "items": [{
        "name": "Unexpected expenses", "description": "Buffer for last-minute costs.",
        "estimated_price": _floor10(alloc["contingency"]), "quantity": 1,
        "search_terms": "party supplies"}]})

    if at_home:
        venues = [{"name": "Your home / terrace", "type": "Residential", "capacity": guests,
                   "estimated_cost": 0, "search_terms": "home party setup ideas"}]
    else:
        venue_budget = _floor10(alloc.get("venue", 0))
        venues = [
            {"name": breakdown[0]["items"][0]["name"], "type": "Event venue", "capacity": guests,
             "estimated_cost": venue_budget, "search_terms": f"party hall for {guests} guests"},
            {"name": "Hotel party / banquet room", "type": "Hotel", "capacity": guests,
             "estimated_cost": venue_budget, "search_terms": "hotel banquet hall"},
        ]
    suggestions = [
        "Book the venue and caterer early - prices rise close to the date.",
        "Ask caterers for a tasting and a per-plate quote that includes service charges.",
        "Keep the contingency amount aside for extra guests or transport.",
    ]
    if inp.needs_catering and alloc.get("catering", 0) / guests < 150:
        suggestions.insert(0, "The food budget per guest is low - consider a potluck or home-cooked menu.")
    return {"budget_breakdown": breakdown, "venue_suggestions": venues, "additional_suggestions": suggestions}


def get_party_recommendations(inp: PartyBudgetInput) -> Dict[str, Any]:
    return _run(
        lambda: [_party_prompt(inp)],
        lambda raw: finalize_breakdown(raw, inp.total_budget, _party_platforms),
        lambda: _party_fallback(inp),
    )


# ---------- Jewelry planner ----------

JEWELRY_JSON_FORMAT = """{
  "outfit_analysis": {"colors": [""], "style": "", "formality": ""},
  "jewelry_recommendations": [
    {"item_type": "", "description": "", "style": "", "estimated_price": 0, "search_terms": ""}
  ],
  "styling_tips": [""]
}"""


def _jewelry_prompt(inp: JewelryBudgetInput, has_image: bool) -> str:
    image_part = (
        "An image of the outfit is attached. Analyse its colors, style and formality, and suggest "
        "jewelry that complements it, considering color, design and occasion appropriateness."
        if has_image else
        'No outfit image was provided; set "outfit_analysis" to null.'
    )
    return f"""
I need jewelry recommendations for India with a total budget of ₹{inp.total_budget:,.2f}.

Occasion: {inp.occasion}
Preferences: {inp.preferences or "Not specified"}

{image_part}

Suggest 3 to 5 pieces available from Amazon, Flipkart, BlueStone, Tanishq, CaratLane or Melorra.
Choose materials that realistically fit the budget (imitation, silver, gold-plated or gold).
"estimated_price" is in INR. The sum of all estimated_price values MUST NOT exceed ₹{inp.total_budget:,.2f}.

Respond only with JSON in this structure:
{JEWELRY_JSON_FORMAT}
"""


NAMED_COLORS = {
    "black": (20, 20, 20), "white": (240, 240, 240), "grey": (128, 128, 128), "red": (200, 30, 40),
    "maroon": (120, 20, 30), "pink": (240, 150, 180), "orange": (240, 130, 30), "yellow": (240, 210, 50),
    "green": (40, 150, 60), "olive": (110, 120, 40), "blue": (40, 80, 200), "navy": (20, 30, 90),
    "teal": (20, 130, 130), "purple": (120, 50, 150), "beige": (220, 200, 160), "brown": (120, 80, 40),
}
WARM_COLORS = {"red", "maroon", "orange", "yellow", "green", "olive", "brown", "beige"}


def analyze_outfit_colors(image_path: str) -> List[str]:
    """Return the main colour names in an image (simple offline analysis)."""
    with Image.open(image_path) as img:
        small = img.convert("RGB").resize((64, 64))
    counts: Counter = Counter()
    for pixel in small.getdata():
        name = min(NAMED_COLORS, key=lambda n: sum((a - b) ** 2 for a, b in zip(pixel, NAMED_COLORS[n])))
        counts[name] += 1
    total = sum(counts.values())
    return [name for name, count in counts.most_common(3) if count / total >= 0.08]


JEWELRY_SETS = {
    "traditional": [("Necklace set", 0.5, "necklace set"), ("Jhumka earrings", 0.3, "jhumka earrings"),
                    ("Bangles set", 0.2, "bangles set")],
    "modern": [("Pendant with chain", 0.45, "pendant necklace"), ("Stud earrings", 0.3, "stud earrings"),
               ("Bracelet", 0.25, "bracelet for women")],
    "men": [("Chain", 0.5, "chain for men"), ("Ring", 0.3, "ring for men"), ("Kada / bracelet", 0.2, "kada for men")],
}
TRADITIONAL_WORDS = ["wedding", "engagement", "festival", "pooja", "puja", "traditional", "ethnic",
                     "saree", "sari", "lehenga", "diwali", "temple"]
MEN_WORDS = ["men", "male", "groom", "him", "boy"]


def _material(budget_per_piece: float) -> str:
    if budget_per_piece < 1500:
        return "oxidised / imitation"
    if budget_per_piece < 8000:
        return "925 sterling silver"
    if budget_per_piece < 40000:
        return "gold-plated silver or lightweight 18K gold"
    return "22K gold"


def _jewelry_fallback(inp: JewelryBudgetInput, image_path: Optional[str]) -> Dict[str, Any]:
    text = f"{inp.occasion} {inp.preferences or ''}".lower()
    words = set(re.findall(r"[a-z]+", text))
    if words & set(MEN_WORDS):
        key = "men"
    elif any(w in text for w in TRADITIONAL_WORDS):
        key = "traditional"
    else:
        key = "modern"
    style = {"men": "classic", "traditional": "traditional", "modern": "modern minimal"}[key]

    items = []
    for item_type, share, terms in JEWELRY_SETS[key]:
        price = _floor10(inp.total_budget * share * 0.9)
        material = _material(price)
        items.append({
            "item_type": item_type,
            "description": f"{material.capitalize()} {item_type.lower()} suited to a {inp.occasion.lower()}.",
            "style": style, "estimated_price": price,
            "search_terms": f"{material.split(' /')[0].split(' or ')[0]} {terms}",
        })

    result: Dict[str, Any] = {"jewelry_recommendations": items}
    tips = [f"Keep one statement piece and let the others stay subtle for a {inp.occasion.lower()}."]
    if image_path:
        try:
            colors = analyze_outfit_colors(image_path)
        except OSError:
            colors = []
        if colors:
            result["outfit_analysis"] = {"colors": colors, "style": "", "formality": ""}
            metal = "gold" if WARM_COLORS & set(colors) else "silver / white gold"
            tips.insert(0, f"Your outfit is mostly {', '.join(colors)} - {metal} tones will complement it.")
    tips.append("Buy from BIS-hallmarked sellers when choosing gold, and check return policies online.")
    result["styling_tips"] = tips
    return result


def get_jewelry_recommendations(inp: JewelryBudgetInput, image_path: Optional[str] = None) -> Dict[str, Any]:
    def contents() -> List[Any]:
        prompt = _jewelry_prompt(inp, has_image=bool(image_path))
        if image_path:
            with Image.open(image_path) as img:
                return [prompt, img.convert("RGB")]
        return [prompt]

    return _run(
        contents,
        lambda raw: finalize_jewelry(raw, inp.total_budget),
        lambda: _jewelry_fallback(inp, image_path),
    )
