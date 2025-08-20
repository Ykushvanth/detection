# api_app.py

import os
import shutil
import requests
from io import BytesIO
from typing import Optional
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
os.makedirs("reference_images", exist_ok=True)

def download_image(url: str, save_path: str):
    response = requests.get(url)
    response.raise_for_status()
    with open(save_path, "wb") as f:
        f.write(response.content)

@app.post("/upload-reference/")
async def upload_reference(file: UploadFile = File(...)):
    ref_path = f"reference_images/{file.filename}"
    with open(ref_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"status": "Reference image uploaded", "filename": file.filename}

@app.post("/detect-from-url/")
async def detect_from_url(image_data: ImageURL):
    try:
        # Download the input image
        detect_path = "detect.jpg"
        download_image(image_data.url, detect_path)

        # Get all unknown persons from Supabase
        response = supabase.table("unknown_persons").select("*").execute()
        unknown_persons = response.data

        # Check against all images from unknown_persons table
        for person in unknown_persons:
            # Download the reference image from Supabase
            ref_image_url = person["image_url"]
            ref_path = f"reference_images/temp_ref.jpg"
            download_image(ref_image_url, ref_path)

            try:
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
                print(f"Error processing image: {str(e)}")
                continue

            finally:
                # Clean up the temporary reference image
                if os.path.exists(ref_path):
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
        if os.path.exists(detect_path):
            os.remove(detect_path)