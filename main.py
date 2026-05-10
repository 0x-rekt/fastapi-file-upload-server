import os
import re
import io
import json
import httpx

from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

import pypdf
from imagekitio import ImageKit
from google import genai
from google.genai import types

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
IMAGEKIT_PUBLIC_KEY = os.getenv("IMAGEKIT_PUBLIC_KEY")
IMAGEKIT_PRIVATE_KEY = os.getenv("IMAGEKIT_PRIVATE_KEY")
IMAGEKIT_URL_ENDPOINT = os.getenv("IMAGEKIT_URL_ENDPOINT")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set")

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

imagekit = ImageKit(private_key=IMAGEKIT_PRIVATE_KEY)

app = FastAPI(title="Resume Analyzer API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://*.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SYSTEM_PROMPT = """You are an advanced AI Resume Analyzer Agent.

Your task is to evaluate a candidate's resume and return a detailed analysis in the following structured JSON schema format.

The schema must match the layout and structure of a visual UI that includes overall score, section scores, summary feedback, improvement tips, strengths, and weaknesses.

📤 INPUT: I will provide a plain text resume.

🎯 GOAL: Output a JSON report as per the schema below. The report should reflect:

overall_score (0–100)

overall_feedback (short message e.g., "Excellent", "Needs improvement")

summary_comment (1–2 sentence evaluation summary)

Section scores for:

Contact Info

Experience

Education

Skills

Each section should include:

score (as percentage)

Optional comment about that section

Tips for improvement (3–5 tips)

What's Good (1–3 strengths)

Needs Improvement (1–3 weaknesses)

🧠 Output JSON Schema:

{
  "overall_score": 85,
  "overall_feedback": "Excellent!",
  "summary_comment": "Your resume is strong, but there are areas to refine.",
  "sections": {
    "contact_info": {
      "score": 95,
      "comment": "Perfectly structured and complete."
    },
    "experience": {
      "score": 88,
      "comment": "Strong bullet points and impact."
    },
    "education": {
      "score": 70,
      "comment": "Consider adding relevant coursework."
    },
    "skills": {
      "score": 60,
      "comment": "Expand on specific skill proficiencies."
    }
  },
  "tips_for_improvement": [
    "Add more numbers and metrics to your experience section to show impact.",
    "Integrate more industry-specific keywords relevant to your target roles.",
    "Start bullet points with strong action verbs to make your achievements stand out."
  ],
  "whats_good": [
    "Clean and professional formatting.",
    "Clear and concise contact information.",
    "Relevant work experience."
  ],
  "needs_improvement": [
    "Skills section lacks detail.",
    "Some experience bullet points could be stronger.",
    "Missing a professional summary/objective."
  ]
}
"""

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract all text from a PDF using pypdf (no C extensions needed)."""
    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    pages_text = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages_text.append(text)
    full_text = "\n".join(pages_text).strip()
    if not full_text:
        raise ValueError("Could not extract any text from the PDF. "
                         "The file may be image-based (scanned). "
                         "Please upload a text-based PDF.")
    return full_text


def analyze_resume(text: str) -> dict:
    """Call Gemini and parse the JSON response."""
    response = gemini_client.models.generate_content(
        model="gemini-3-flash-preview",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
        ),
        contents=text,
    )

    candidate_text: str = ""
    try:
        candidate_text = response.candidates[0].content.parts[0].text
    except (AttributeError, IndexError, TypeError) as exc:
        raise ValueError("No response from AI model") from exc

    if not candidate_text:
        raise ValueError("Empty response from AI model")

    # Strip markdown code fences if present
    raw_json = re.sub(r"```json\n?", "", candidate_text)
    raw_json = re.sub(r"```\n?", "", raw_json).strip()

    # Extract the first JSON object
    json_match = re.search(r"\{[\s\S]*\}", raw_json)
    if json_match:
        raw_json = json_match.group(0)

    try:
        return json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON from AI model. Received: {candidate_text[:100]}"
        ) from exc


def upload_to_imagekit(file_bytes: bytes, filename: str) -> str:
    """Upload a file to ImageKit and return its CDN URL.

    imagekitio v5 API:
      - Upload is on the `.files` sub-resource
      - The `file` parameter must be bytes, an io.IOBase instance, or a PathLike.
        A base64-encoded string is NOT accepted by the SDK.
      - CDN URL = url_endpoint.rstrip('/') + '/' + result.file_path.lstrip('/')
    """
    result = imagekit.files.upload(
        file=io.BytesIO(file_bytes),
        file_name=filename,
    )
    # Build the public CDN URL from the endpoint + the path returned by the API
    endpoint = (IMAGEKIT_URL_ENDPOINT or "").rstrip("/")
    file_path = getattr(result, "file_path", None) or getattr(result, "filePath", filename)
    return f"{endpoint}/{file_path.lstrip('/')}"

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze-resume")
async def analyze_resume_endpoint(
    resume: UploadFile = File(..., description="PDF resume file"),
):
    """
    Parse a PDF resume, run Gemini analysis, upload to ImageKit, and return
    the structured analysis together with the hosted PDF URL.
    """
    if resume.content_type not in ("application/pdf", "application/octet-stream"):
        if not (resume.filename or "").lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are accepted.",
            )

    try:
        pdf_bytes = await resume.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read file: {exc}") from exc

    # --- extract text ---
    try:
        text = extract_text_from_pdf(pdf_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"PDF parsing error: {exc}") from exc

    # --- AI analysis ---
    try:
        analysis = analyze_resume(text)
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"AI analysis error: {exc}") from exc

    # --- upload PDF to ImageKit ---
    try:
        url = upload_to_imagekit(pdf_bytes, resume.filename or "resume.pdf")
    except Exception as exc:
        # Upload failure is non-fatal; return analysis without URL
        url = None
        print(f"[IMAGEKIT_UPLOAD_ERROR] {exc}")

    return JSONResponse(content={"analysis": analysis, "url": url})
