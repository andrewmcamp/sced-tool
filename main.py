from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional
from contextlib import asynccontextmanager
import lancedb
from sentence_transformers import SentenceTransformer

# --- Configuration ---
DB_PATH = "./sced_lancedb"
TABLE_NAME = "sced_courses"

# --- Global State ---
ml_resources = {}


# --- Lifespan Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading Embedding Model...")
    ml_resources["model"] = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"Connecting to Database at {DB_PATH}...")
    db = lancedb.connect(DB_PATH)
    ml_resources["table"] = db.open_table(TABLE_NAME)
    print("Server Ready!")
    yield
    print("Shutting down...")
    ml_resources.clear()


# --- Initialize App ---
app = FastAPI(title="SCED Course Search API", lifespan=lifespan)

# --- Template Configuration ---
# This points to the "templates" folder
templates = Jinja2Templates(directory="templates")


# --- Data Models ---
class SearchRequest(BaseModel):
    subject: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None


class SearchResult(BaseModel):
    rank: int
    score: float
    subject: str
    course_code: str
    title: str
    description_snippet: str


# --- Helper Function for Search Logic ---
# We extract this so both the API and the UI can use the exact same logic
def perform_search_logic(subject, title, description):
    model = ml_resources.get("model")
    table = ml_resources.get("table")

    if not model or not table:
        raise HTTPException(status_code=500, detail="Server resources not initialized.")

    query_text = ""
    sql_filter = None

    # Logic for input combinations
    if subject and description:
        # Scenario 3: Subject + Description
        query_text = description
        sql_filter = f"subject = '{subject}'"
    elif subject and title:
        # Scenario 2: Subject + Title
        query_text = title
        sql_filter = f"subject = '{subject}'"
    elif description:
        # Scenario 1: Description Only
        query_text = description
        sql_filter = None
    else:
        return []  # Return empty if invalid input

    # Execute Search
    query_vector = model.encode(query_text).tolist()
    search_op = table.search(query_vector).limit(10)

    if sql_filter:
        search_op = search_op.where(sql_filter)

    results_df = search_op.to_pandas()

    if results_df.empty:
        return []

    # Format Results
    results = []
    for idx, row in results_df.iterrows():
        results.append(
            {
                "rank": idx + 1,
                "score": round(row["_distance"], 3),
                "subject": str(row["subject"]),
                "course_code": str(row["sced_course_code"]),
                "title": str(row["sced_course_title"]),
                "description_snippet": str(row["sced_course_description"])[:200]
                + "...",
            }
        )
    return results


# --- API Endpoint (JSON) ---
@app.post("/search", response_model=List[SearchResult])
def search_api(request: SearchRequest):
    results = perform_search_logic(request.subject, request.title, request.description)
    if not results and not (request.description or (request.subject and request.title)):
        raise HTTPException(status_code=400, detail="Invalid search parameters.")
    return results


# --- UI Routes (HTML) ---
@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    # Render the empty form
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/", response_class=HTMLResponse)
def submit_form(
    request: Request,
    subject: str = Form(None),
    title: str = Form(None),
    description: str = Form(None),
):
    # Clean inputs (convert empty strings to None)
    s = subject.strip() if subject else None
    t = title.strip() if title else None
    d = description.strip() if description else None

    # Run logic
    results = perform_search_logic(s, t, d)

    # Render template with results
    return templates.TemplateResponse(
        "index.html", {"request": request, "results": results, "submitted": True}
    )
