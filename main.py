from fastapi import FastAPI, HTTPException, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional
from contextlib import asynccontextmanager
import lancedb
from sentence_transformers import SentenceTransformer
import pandas as pd
import io

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
def perform_search_logic(subject, title, description):
    model = ml_resources.get("model")
    table = ml_resources.get("table")

    if not model or not table:
        raise HTTPException(status_code=500, detail="Server resources not initialized.")

    query_text = ""
    sql_filter = None

    # Handle empty strings/NaNs safely
    subject = subject if subject and str(subject).lower() != 'nan' else None
    title = title if title and str(title).lower() != 'nan' else None
    description = description if description and str(description).lower() != 'nan' else None

    # Logic for input combinations
    if subject and description:
        query_text = description
        sql_filter = f"subject = '{subject}'"
    elif subject and title:
        query_text = title
        sql_filter = f"subject = '{subject}'"
    elif description:
        query_text = description
        sql_filter = None
    elif title: 
        # Added fallback: if only title is provided (common in CSVs)
        query_text = title
        sql_filter = None
    else:
        return []

    # Execute Search
    query_vector = model.encode(query_text).tolist()
    
    # We limit to 5 per specific query to keep things fast
    search_op = table.search(query_vector).limit(5)

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
                "description_snippet": str(row["sced_course_description"])[:200] + "...",
            }
        )
    return results

# --- API Endpoint (JSON) ---
@app.post("/search", response_model=List[SearchResult])
def search_api(request: SearchRequest):
    results = perform_search_logic(request.subject, request.title, request.description)
    return results

# --- UI Routes (HTML) ---
@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/search", response_class=HTMLResponse) # Changed from "/" to "/search" to match form action
def submit_single_search(
    request: Request,
    subject: str = Form(None),
    title: str = Form(None),
    description: str = Form(None),
):
    s = subject.strip() if subject else None
    t = title.strip() if title else None
    d = description.strip() if description else None

    results = perform_search_logic(s, t, d)

    return templates.TemplateResponse(
        "index.html", {"request": request, "results": results, "submitted": True}
    )

# --- NEW: Batch Upload Endpoint ---
@app.post("/upload", response_class=HTMLResponse)
async def upload_file(
    request: Request, 
    file: UploadFile = File(...), 
    strict_match: Optional[str] = Form(None) # Checkboxes return 'on' or None
):
    if not file.filename.endswith('.csv'):
        # In a real app, handle error gracefully
        return templates.TemplateResponse("index.html", {"request": request, "results": [], "submitted": True})

    # Read CSV content
    content = await file.read()
    
    try:
        # Use Pandas to parse the CSV bytes
        df = pd.read_csv(io.BytesIO(content))
        
        all_results = []
        
        # Iterate through rows
        for _, row in df.iterrows():
            # Flexible column matching
            # We look for columns named 'Title', 'Description', 'Subject' (case insensitive)
            row_data = {k.lower(): v for k, v in row.items()}
            
            t = row_data.get('title')
            d = row_data.get('description')
            s = row_data.get('subject')
            
            # Run search for this row
            matches = perform_search_logic(s, t, d)
            
            # Logic: If matches found, take the TOP result and add to our display list
            # (You could change this to add all matches if you prefer)
            if matches:
                # We tag the result with the input title so the user knows what it matched
                best_match = matches[0]
                # Optional: Prepend "Matched for: [Input Title]" to description for clarity in UI
                if t:
                    best_match['description_snippet'] = f"[Input: {t}] " + best_match['description_snippet']
                
                all_results.append(best_match)

        return templates.TemplateResponse(
            "index.html", 
            {"request": request, "results": all_results, "submitted": True}
        )

    except Exception as e:
        print(f"Error processing CSV: {e}")
        return templates.TemplateResponse("index.html", {"request": request, "results": []})
