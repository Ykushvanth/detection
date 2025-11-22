---
title: Face Detection API
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Face Detection API

A FastAPI-based face detection and verification service using DeepFace and Supabase integration.

## Features

- **Face Verification**: Verify faces against a database of unknown persons
- **Reference Image Upload**: Upload reference images for comparison
- **URL-based Detection**: Detect faces from image URLs
- **Supabase Integration**: Stores and retrieves person data from Supabase

## API Endpoints

### `GET /`
Returns API information and available endpoints.

### `GET /health`
Health check endpoint.

### `POST /upload-reference/`
Upload a reference image file.

**Request**: `multipart/form-data` with `file` field

**Response**:
```json
{
  "status": "Reference image uploaded",
  "filename": "image.jpg"
}
```

### `POST /detect-from-url/`
Detect and verify a face from an image URL.

**Request Body**:
```json
{
  "url": "https://example.com/image.jpg"
}
```

**Response**:
```json
{
  "verified": true,
  "matched_with": {
    "id": 1,
    "name": "John Doe",
    "image_url": "https://..."
  },
  "distance": 0.23,
  "threshold": 0.68
}
```

## Environment Variables

The following environment variables are used:

- `PORT`: Server port (default: 7860)
- Supabase credentials are configured in the code

## Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python index.py
```

Or using uvicorn directly:
```bash
uvicorn index:app --host 0.0.0.0 --port 7860
```

## Docker Deployment

### Build the Docker image:
```bash
docker build -t face-detection-api .
```

### Run the container:
```bash
docker run -p 7860:7860 face-detection-api
```

## Hugging Face Spaces Deployment

This repository is configured for deployment on Hugging Face Spaces using Docker.

1. Push this repository to Hugging Face Spaces
2. The Dockerfile will be automatically used to build and deploy the application
3. The API will be available at the Space's URL

## Technologies Used

- FastAPI
- DeepFace
- Supabase
- OpenCV
- TensorFlow
- Uvicorn



