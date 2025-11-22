# # api_app.py

# import os
# import shutil
# import requests
# from io import BytesIO
# from typing import Optional
# from fastapi import FastAPI, UploadFile, File
# from fastapi.responses import JSONResponse
# from deepface import DeepFace
# from supabase import create_client, Client
# from pydantic import BaseModel

# # Initialize Supabase client
# supabase_url = "https://nrxqcfdbyscqgrdrqegu.supabase.co"
# supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5yeHFjZmRieXNjcWdyZHJxZWd1Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTIwMzQzNTIsImV4cCI6MjA2NzYxMDM1Mn0.TR9RdSYoaKwryNAJRlD6rhas4ri3liqT4p2-yvE6Vtg"
# supabase: Client = create_client(supabase_url, supabase_key)

# class ImageURL(BaseModel):
#     url: str

# app = FastAPI()
# os.makedirs("reference_images", exist_ok=True)

# def download_image(url: str, save_path: str):
#     response = requests.get(url)
#     response.raise_for_status()
#     with open(save_path, "wb") as f:
#         f.write(response.content)

# @app.post("/upload-reference/")
# async def upload_reference(file: UploadFile = File(...)):
#     ref_path = f"reference_images/{file.filename}"
#     with open(ref_path, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)
#     return {"status": "Reference image uploaded", "filename": file.filename}

# @app.post("/detect-from-url/")
# async def detect_from_url(image_data: ImageURL):
#     try:
#         # Download the input image
#         detect_path = "detect.jpg"
#         download_image(image_data.url, detect_path)

#         # Get all unknown persons from Supabase
#         response = supabase.table("unknown_persons").select("*").execute()
#         unknown_persons = response.data

#         # Check against all images from unknown_persons table
#         for person in unknown_persons:
#             # Download the reference image from Supabase
#             ref_image_url = person["image_url"]
#             ref_path = f"reference_images/temp_ref.jpg"
#             download_image(ref_image_url, ref_path)

#             try:
#                 result = DeepFace.verify(
#                     img1_path=ref_path,
#                     img2_path=detect_path,
#                     model_name="ArcFace",
#                     detector_backend="opencv",  # Changed from retinaface to opencv
#                     enforce_detection=False
#                 )

#                 if result["verified"]:
#                     return {
#                         "verified": True,
#                         "matched_with": person,
#                         "distance": result["distance"],
#                         "threshold": result["threshold"]
#                     }

#             except Exception as e:
#                 print(f"Error processing image: {str(e)}")
#                 continue

#             finally:
#                 # Clean up the temporary reference image
#                 if os.path.exists(ref_path):
#                     os.remove(ref_path)

#         return {
#             "verified": False,
#             "matched_with": None
#         }

#     except Exception as e:
#         return JSONResponse(
#             status_code=500,
#             content={"error": f"Failed to process image: {str(e)}"}
#         )

#     finally:
#         # Clean up the detect image
#         if os.path.exists(detect_path):
#             os.remove(detect_path)




import os
import tempfile
import requests
import logging
from io import BytesIO
from typing import Optional

# Import tf_keras before TensorFlow to enable legacy Keras support
# This must be done before importing deepface which uses tf.keras
try:
    import tf_keras
    os.environ['TF_USE_LEGACY_KERAS'] = '1'
except ImportError:
    pass

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from deepface import DeepFace
from supabase import create_client, Client
from pydantic import BaseModel, HttpUrl

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Supabase client from environment variables
# Default values for local testing (replace with your actual values or set via environment variables)
supabase_url = os.getenv("SUPABASE_URL", "https://nrxqcfdbyscqgrdrqegu.supabase.co")
supabase_key = os.getenv("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5yeHFjZmRieXNjcWdyZHJxZWd1Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTIwMzQzNTIsImV4cCI6MjA2NzYxMDM1Mn0.TR9RdSYoaKwryNAJRlD6rhas4ri3liqT4p2-yvE6Vtg")

if not supabase_url or not supabase_key:
    raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set")

supabase: Client = create_client(supabase_url, supabase_key)

class ImageURL(BaseModel):
    url: HttpUrl

app = FastAPI(title="Face Detection API", version="1.0.0")
os.makedirs("reference_images", exist_ok=True)

# Configuration
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
VERIFICATION_THRESHOLD = 0.68  # ArcFace default threshold

@app.get("/")
async def root():
    return {
        "message": "Face Detection API",
        "endpoints": {
            "health": "/health",
            "upload_reference": "/upload-reference/",
            "detect_from_url": "/detect-from-url/",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}

def download_image(url: str, max_size: int = MAX_FILE_SIZE) -> bytes:
    """Download image with size validation"""
    try:
        response = requests.get(url, timeout=10, stream=True)
    response.raise_for_status()
        
        # Check content length
        content_length = response.headers.get('content-length')
        if content_length and int(content_length) > max_size:
            raise HTTPException(400, "Image too large")
        
        # Download with size limit
        content = BytesIO()
        downloaded = 0
        for chunk in response.iter_content(chunk_size=8192):
            downloaded += len(chunk)
            if downloaded > max_size:
                raise HTTPException(400, "Image too large")
            content.write(chunk)
        
        return content.getvalue()
    
    except requests.RequestException as e:
        logger.error(f"Failed to download image: {str(e)}")
        raise HTTPException(400, f"Failed to download image: {str(e)}")

@app.post("/upload-reference/")
async def upload_reference(file: UploadFile = File(...)):
    """Upload a reference image for face detection"""
    
    # Validate file extension
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Invalid file type. Allowed: {ALLOWED_EXTENSIONS}")
    
    # Validate file size
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(400, "File too large")
    
    # Save file
    ref_path = f"reference_images/{file.filename}"
    try:
    with open(ref_path, "wb") as buffer:
            buffer.write(contents)
    return {"status": "Reference image uploaded", "filename": file.filename}
    except Exception as e:
        logger.error(f"Failed to save file: {str(e)}")
        raise HTTPException(500, "Failed to save file")

@app.post("/detect-from-url/")
async def detect_from_url(image_data: ImageURL):
    """Detect faces in an image from URL and match against database"""
    
    detect_fd = None
    ref_fd = None
    
    try:
        # Download the input image
        logger.info(f"Downloading input image from {image_data.url}")
        input_image_data = download_image(str(image_data.url))
        
        # Save to temporary file
        detect_fd = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        detect_fd.write(input_image_data)
        detect_fd.close()
        detect_path = detect_fd.name

        # Get all unknown persons from Supabase
        logger.info("Fetching unknown persons from database")
        response = supabase.table("unknown_persons").select("*").execute()
        unknown_persons = response.data
        
        if not unknown_persons:
            logger.warning("No unknown persons in database")
            return {"verified": False, "matched_with": None, "message": "No reference images in database"}
        
        logger.info(f"Checking against {len(unknown_persons)} reference images")

        # Check against all images from unknown_persons table
        for idx, person in enumerate(unknown_persons):
            try:
            # Download the reference image from Supabase
            ref_image_url = person["image_url"]
                logger.info(f"Processing reference {idx + 1}/{len(unknown_persons)}")
                
                ref_image_data = download_image(ref_image_url)
                
                # Save to temporary file
                ref_fd = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                ref_fd.write(ref_image_data)
                ref_fd.close()
                ref_path = ref_fd.name
                
                # Perform face verification
                result = DeepFace.verify(
                    img1_path=ref_path,
                    img2_path=detect_path,
                    model_name="ArcFace",
                    detector_backend="retinaface",  # Using retinaface for better accuracy
                    enforce_detection=False
                )
                
                logger.info(f"Verification result: distance={result['distance']}, threshold={result['threshold']}")

                if result["verified"]:
                    logger.info(f"Match found with person ID: {person.get('id')}")
                    return {
                        "verified": True,
                        "matched_with": person,
                        "distance": result["distance"],
                        "threshold": result["threshold"],
                        "confidence": 1 - (result["distance"] / result["threshold"])
                    }

            except Exception as e:
                logger.error(f"Error processing reference image {idx + 1}: {str(e)}")
                continue

            finally:
                # Clean up the temporary reference image
                if ref_fd and os.path.exists(ref_fd.name):
                    try:
                        os.unlink(ref_fd.name)
                    except:
                        pass
                ref_fd = None
        
        logger.info("No match found")
        return {
            "verified": False,
            "matched_with": None,
            "message": f"No match found among {len(unknown_persons)} reference images"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(500, f"Failed to process image: {str(e)}")

    finally:
        # Clean up the detect image
        if detect_fd and os.path.exists(detect_fd.name):
            try:
                os.unlink(detect_fd.name)
            except:
                pass