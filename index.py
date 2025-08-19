# api_app.py

import os
import shutil
import requests
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from deepface import DeepFace
from supabase import create_client, Client
from pydantic import BaseModel

# ----------------- Supabase Setup -----------------
SUPABASE_URL = "https://nrxqcfdbyscqgrdrqegu.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5yeHFjZmRieXNjcWdyZHJxZWd1Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTIwMzQzNTIsImV4cCI6MjA2NzYxMDM1Mn0.TR9RdSYoaKwryNAJRlD6rhas4ri3liqT4p2-yvE6Vtg"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ----------------- FastAPI Setup -----------------
app = FastAPI()
os.makedirs("reference_images", exist_ok=True)

class ImageURL(BaseModel):
    url: str

# ----------------- Utility Functions -----------------
def download_image(url: str, save_path: str):
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    with open(save_path, "wb") as f:
        f.write(response.content)

# ----------------- Endpoints -----------------
@app.post("/upload-reference/")
async def upload_reference(file: UploadFile = File(...)):
    ref_path = f"reference_images/{file.filename}"
    with open(ref_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"status": "Reference image uploaded", "filename": file.filename}

@app.post("/detect-from-url/")
async def detect_from_url(image_data: ImageURL):
    detect_path = "detect.jpg"
    try:
        # Download the input image
        download_image(image_data.url, detect_path)

        # Fetch all unknown persons from Supabase
        response = supabase.table("unknown_persons").select("*").execute()
        unknown_persons = response.data

        for person in unknown_persons:
            ref_image_url = person.get("image_url")
            if not ref_image_url:
                continue

            ref_path = "reference_images/temp_ref.jpg"
            download_image(ref_image_url, ref_path)

            try:
                result = DeepFace.verify(
                    img1_path=ref_path,
                    img2_path=detect_path,
                    model_name="ArcFace",
                    detector_backend="retinaface",
                    enforce_detection=False
                )

                if result.get("verified"):
                    return {
                        "verified": True,
                        "matched_with": person,
                        "distance": result.get("distance"),
                        "threshold": result.get("threshold")
                    }

            except Exception as e:
                print(f"Error verifying image: {str(e)}")
                continue

            finally:
                if os.path.exists(ref_path):
                    os.remove(ref_path)

        return {"verified": False, "matched_with": None}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    finally:
        if os.path.exists(detect_path):
            os.remove(detect_path)

# ----------------- Run on Render -----------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # Render dynamically sets PORT
    uvicorn.run("index:app", host="0.0.0.0", port=port, reload=False)
