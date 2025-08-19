import os
import shutil
import requests
import tempfile
import uuid
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from deepface import DeepFace
from supabase import create_client, Client
from pydantic import BaseModel

# Initialize Supabase client
supabase_url = "https://nrxqcfdbyscqgrdrqegu.supabase.co"
supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5yeHFjZmRieXNjcWdyZHJxZWd1Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTIwMzQzNTIsImV4cCI6MjA2NzYxMDM1Mn0.TR9RdSYoaKwryNAJRlD6rhas4ri3liqT4p2-yvE6Vtg"
supabase: Client = create_client(supabase_url, supabase_key)

class ImageURL(BaseModel):
    url: str

app = FastAPI()

# Root endpoint for health check
@app.get("/")
async def root():
    return {"message": "Face Recognition API is running", "status": "healthy"}

def download_image(url: str, save_path: str):
    response = requests.get(url)
    response.raise_for_status()
    with open(save_path, "wb") as f:
        f.write(response.content)

@app.post("/upload-reference/")
async def upload_reference(file: UploadFile = File(...)):
    try:
        # Use temporary file instead of persistent storage
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name
        # Here you could process the image and store metadata in Supabase
        # For now, just acknowledge the upload
        # Clean up the temporary file
        os.remove(temp_path)
        return {"status": "Reference image processed", "filename": file.filename}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to process reference image: {str(e)}"}
        )

@app.post("/detect-from-url/")
async def detect_from_url(image_data: ImageURL):
    detect_path = None
    try:
        # Download the input image with unique filename
        detect_path = f"temp_detect_{uuid.uuid4().hex}.jpg"
        download_image(image_data.url, detect_path)
        # Get all unknown persons from Supabase
        response = supabase.table("unknown_persons").select("*").execute()
        unknown_persons = response.data
        # Check against all images from unknown_persons table
        for person in unknown_persons:
            ref_path = None
            try:
                # Download the reference image from Supabase with unique filename
                ref_image_url = person["image_url"]
                ref_path = f"temp_ref_{uuid.uuid4().hex}.jpg"
                download_image(ref_image_url, ref_path)
                result = DeepFace.verify(
                    img1_path=ref_path,
                    img2_path=detect_path,
                    model_name="ArcFace",
                    detector_backend="retinaface",
                    enforce_detection=False
                )
                if result["verified"]:
                    return {
                        "verified": True,
                        "matched_with": person,
                        "distance": result["distance"],
                        "threshold": result["threshold"]
                    }
            except Exception as e:
                print(f"Error processing image for person {person.get('id', 'unknown')}: {str(e)}")
                continue
            finally:
                # Clean up the temporary reference image
                if ref_path and os.path.exists(ref_path):
                    os.remove(ref_path)
        return {
            "verified": False,
            "matched_with": None
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to process image: {str(e)}"}
        )
    finally:
        # Clean up the detect image
        if detect_path and os.path.exists(detect_path):
            os.remove(detect_path)


# Server configuration for Render
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
