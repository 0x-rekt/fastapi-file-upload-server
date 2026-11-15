import os
import re
import io
import json
import httpx
from datetime import date

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

import pypdf
from imagekitio import ImageKit
from google import genai
from google.genai import errors, types

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
IMAGEKIT_PUBLIC_KEY = os.getenv("IMAGEKIT_PUBLIC_KEY")
IMAGEKIT_PRIVATE_KEY = os.getenv("IMAGEKIT_PRIVATE_KEY")
IMAGEKIT_URL_ENDPOINT = os.getenv("IMAGEKIT_URL_ENDPOINT")

GEMINI_MODEL = "gemini-3.8-flash"
# Used only when the primary model is overloaded (429/5xx after retries).
GEMINI_FALLBACK_MODEL = "gemini-3.6-flash"
GEMINI_STABLE_MODEL = "gemini-2.5-flash"
RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set")

gemini_client = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options=types.HttpOptions(
        retry_options=types.HttpRetryOptions(
            attempts=4,
            initialDelay=1.0,
            maxDelay=8.0,
            expBase=2.0,
            jitter=1.0,
            httpStatusCodes=list(RETRYABLE_STATUS_CODES),
        ),
    ),
)

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

COVER_LETTER_SYSTEM_PROMPT = """You are an advanced AI Cover Letter Generator.

Your task is to write a personalized cover letter based on a candidate's resume and a job description. Return the output as a JSON object following this exact schema:

{
  "intro": string,
  "body": string,
  "closing": string,
  "tone": string,
  "length": string
}

⚠️ Requirements:
- intro is a 1-2 sentence opening greeting.
- body is 2-3 paragraphs explaining fit, referencing resume & job.
- closing is a 1-2 sentence enthusiastic closing.
- tone should reflect the requested tone (e.g., "professional", "enthusiastic").
- length should reflect the approximate requested length (e.g., "~250 words").
- Output valid JSON only—no markdown or extra keys.
- Use information from both resume and job description.
- Extract key achievements from the resumeText automatically.
- Ensure the tone and length match the input preferences.
- Be concise, authentic, and contextually relevant.

📥 INPUT (user message will include):
- resumeText: the candidate's resume content (plain text).
- jobTitle: position title.
- companyName: name of the company.
- jobDescription: full or excerpted job posting.
- tone: desired tone.
- length: approximate desired length.

When given these inputs, analyze and craft a tailored cover letter fitting the schema above. Output JSON only.
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


def _generate_with_fallback(system_prompt: str, contents: str):
    """Call the primary Gemini model, falling back if it is overloaded."""
    config = types.GenerateContentConfig(system_instruction=system_prompt)

    try:
        return gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            config=config,
            contents=contents,
        )
    except errors.APIError as exc:
        if exc.code not in RETRYABLE_STATUS_CODES:
            raise
        print(
            f"[GEMINI_FALLBACK] {GEMINI_MODEL} returned {exc.code}; "
            f"retrying with {GEMINI_FALLBACK_MODEL}"
        )
        try:
            return gemini_client.models.generate_content(
                model=GEMINI_FALLBACK_MODEL,
                config=config,
                contents=contents,
            )
        except errors.APIError as fallback_exc:
            if fallback_exc.code not in RETRYABLE_STATUS_CODES:
                raise
            print(
                f"[GEMINI_FALLBACK] {GEMINI_FALLBACK_MODEL} returned "
                f"{fallback_exc.code}; retrying with {GEMINI_STABLE_MODEL}"
            )
            return gemini_client.models.generate_content(
                model=GEMINI_STABLE_MODEL,
                config=config,
                contents=contents,
            )


def _call_gemini_json(system_prompt: str, contents: str) -> dict:
    """Call Gemini and parse the JSON object from its response."""
    response = _generate_with_fallback(system_prompt, contents)

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


def analyze_resume(text: str) -> dict:
    """Call Gemini and parse the JSON response."""
    date_context = f"""

DATE VALIDATION:
- Today is {date.today().isoformat()}.
- Treat a date or date range as future only when its end date is after today.
- A completed range such as July 2025 - August 2025 is in the past and must not
  be described as a future date.
- Do not flag past employment, education, or project dates as future dates.
- If a date range has an ongoing or "Present" end date, treat it as current,
  not future, unless the stated start date is after today.
"""
    return _call_gemini_json(SYSTEM_PROMPT + date_context, text)


def generate_cover_letter(
    resume_text: str,
    job_title: str,
    company_name: str,
    job_description: str,
    tone: str,
    length: str,
) -> dict:
    """Call Gemini to generate a structured cover letter."""
    llm_input = json.dumps(
        {
            "resumeText": resume_text,
            "jobTitle": job_title,
            "companyName": company_name,
            "jobDescription": job_description,
            "tone": tone,
            "length": length,
        }
    )
    return _call_gemini_json(COVER_LETTER_SYSTEM_PROMPT, llm_input)


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
@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/analyze-resume")
@app.post("/api/analyze-resume")
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


@app.post("/generate-cover-letter")
@app.post("/api/generate-cover-letter")
async def generate_cover_letter_endpoint(
    resume: UploadFile = File(..., description="PDF resume file"),
    job_title: str = Form(..., alias="jobTitle"),
    company_name: str = Form(..., alias="companyName"),
    job_description: str = Form("", alias="jobDescription"),
    tone: str = Form(...),
    length: str = Form(...),
):
    """
    Parse a PDF resume, generate a tailored cover letter with Gemini, upload the
    PDF to ImageKit, and return the structured cover letter with the hosted URL.
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
        resume_text = extract_text_from_pdf(pdf_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"PDF parsing error: {exc}") from exc

    # --- AI cover letter generation ---
    try:
        cover_letter = generate_cover_letter(
            resume_text=resume_text,
            job_title=job_title,
            company_name=company_name,
            job_description=job_description,
            tone=tone,
            length=length,
        )
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"AI cover letter error: {exc}"
        ) from exc

    # --- upload PDF to ImageKit ---
    try:
        url = upload_to_imagekit(pdf_bytes, resume.filename or "resume.pdf")
    except Exception as exc:
        # Upload failure is non-fatal; return the cover letter without URL
        url = None
        print(f"[IMAGEKIT_UPLOAD_ERROR] {exc}")

    return JSONResponse(content={"cover_letter": cover_letter, "pdf_url": url})
