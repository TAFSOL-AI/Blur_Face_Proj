from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import List, Optional
import cv2
import numpy as np
import os
import uuid
import shutil
from pydantic import BaseModel
import logging
import time
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse,FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

templates = Jinja2Templates(directory="templates")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create directories if they don't exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("temp", exist_ok=True)

# Initialize FastAPI app
app = FastAPI(title="Face Blur API")
app.mount("/static", StaticFiles(directory="static/"), name="static")

# Custom middleware for request timing
class TimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        logger.info(f"Request to {request.url.path} took {process_time:.4f} seconds")
        return response

# Add middlewares
app.add_middleware(TimingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
app.mount("/images", StaticFiles(directory="results"), name="images")

# Store image data in memory (in production, use a proper database)
image_data = {}

class BlurRequest(BaseModel):
    image_id: str
    x: int
    y: int
    radius: int = 10

class FreeSelectionBlurRequest(BaseModel):
    image_id: str
    points: List[List[int]]  # List of [x, y] coordinates
    radius: int = 10

class AdjustBlurRequest(BaseModel):
    image_id: str
    radius: int = 10


@app.get("/")
async def get_index():
    return FileResponse("static/index.html")

@app.post("/upload/")
async def upload_images(files: List[UploadFile] = File(...)):
    """Upload multiple images for processing"""
    result = []
    
    for file in files:
        if not file.filename or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Invalid file format. Please upload images only.")
        
        # Generate unique ID for this image
        image_id = str(uuid.uuid4())
        file_extension = os.path.splitext(file.filename)[1]
        file_path = f"uploads/{image_id}{file_extension}"
        
        # Save the file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Open image with OpenCV
        image = cv2.imread(file_path)
        if image is None:
            os.remove(file_path)
            raise HTTPException(status_code=400, detail=f"Could not process image: {file.filename}")
        
        # Detect faces
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
        
        # Store image data
        image_data[image_id] = {
            "original_path": file_path,
            "current_image": image.copy(),
            "original_image": image.copy(),
            "faces": faces.tolist() if len(faces) > 0 else [],
            "filename": file.filename,
            "undo_history": [],
            "file_extension": file_extension,
            "last_blur": None,
            "pre_blur_state": None  # Add this to store the image state before blur
        }
        
        # Save initial version
        result_path = f"results/{image_id}{file_extension}"
        cv2.imwrite(result_path, image)
        
        # Return data about detected faces
        face_list = []
        for (x, y, w, h) in faces:
            face_list.append({"x": int(x), "y": int(y), "width": int(w), "height": int(h)})
        
        result.append({
            "image_id": image_id,
            "original_filename": file.filename,
            "faces_detected": len(faces),
            "faces": face_list,
            "preview_url": f"/images/{image_id}{file_extension}"
        })
    
    return result

@app.post("/blur/face/")
async def blur_face(request: BlurRequest):
    """Blur a face at the specified coordinates"""
    image_id = request.image_id
    
    if image_id not in image_data:
        raise HTTPException(status_code=404, detail="Image not found")
    
    img_data = image_data[image_id]
    image = img_data["current_image"].copy()
    faces = np.array(img_data["faces"])
    
    # Save state before modification
    img_data["undo_history"].append(image.copy())
    
    # Find the target face
    target_face = None
    for (fx, fy, fw, fh) in faces:
        if fx <= request.x <= fx + fw and fy <= request.y <= fy + fh:
            target_face = (fx, fy, fw, fh)
            break
            
    if target_face:
        fx, fy, fw, fh = target_face
        
        # Store the pre-blur state of the face region
        pre_blur_image = image.copy()
        img_data["pre_blur_state"] = {
            "image": pre_blur_image,
            "region": (fx, fy, fw, fh) if target_face else None
        }
        
        # Apply blur to the face
        roi = image[fy:fy+fh, fx:fx+fw]
        blurred = cv2.GaussianBlur(roi, (0, 0), request.radius)
        image[fy:fy+fh, fx:fx+fw] = blurred
    
    # Update current image
    img_data["current_image"] = image

    # Store details about the blur operation
    img_data["last_blur"] = {
        "type": "face",
        "face": target_face,
        "radius": request.radius
    }
    
    # Save result
    result_path = f"results/{image_id}{img_data['file_extension']}"
    cv2.imwrite(result_path, image)
    
    return {"status": "success", "preview_url": f"/images/{image_id}{img_data['file_extension']}"}

@app.post("/blur/selection/")
async def blur_free_selection(request: FreeSelectionBlurRequest):
    """Blur a custom selected area"""
    image_id = request.image_id
    
    if image_id not in image_data:
        raise HTTPException(status_code=404, detail="Image not found")
    
    img_data = image_data[image_id]
    image = img_data["current_image"].copy()
    
    # Save state before modification
    img_data["undo_history"].append(image.copy())
    
    # Convert points to numpy array
    points = np.array(request.points, dtype=np.int32)
    
    if len(points) < 3:
        raise HTTPException(status_code=400, detail="At least 3 points are required for custom selection")
    
    # Create a mask from the points
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [points], 255)
    
    # Store the pre-blur state with the mask
    pre_blur_image = image.copy()
    img_data["pre_blur_state"] = {
        "image": pre_blur_image,
        "mask": mask.copy()
    }
    
    # Create a blurred version of the entire image
    blurred = cv2.GaussianBlur(image, (0, 0), request.radius)
    
    # Blend the original and blurred images using the mask
    mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
    image[:] = image * (1 - mask_3d) + blurred * mask_3d
    
    # Update current image
    img_data["current_image"] = image
    
    # Store details about the blur operation
    img_data["last_blur"] = {
        "type": "custom",
        "points": points.tolist(),
        "radius": request.radius
    }

    # Save result
    result_path = f"results/{image_id}{img_data['file_extension']}"
    cv2.imwrite(result_path, image)
    
    return {"status": "success", "preview_url": f"/images/{image_id}{img_data['file_extension']}"}

@app.post("/adjust-blur/")
async def adjust_last_blur(request: AdjustBlurRequest):
    """Adjust the intensity of the last applied blur"""
    image_id = request.image_id
    
    if image_id not in image_data:
        raise HTTPException(status_code=404, detail="Image not found")
    
    img_data = image_data[image_id]
    
    # Check if there's a last blur operation to adjust
    if not img_data["last_blur"] or not img_data["pre_blur_state"]:
        raise HTTPException(status_code=400, detail="No previous blur operation found")
    
    # Get the last blur operation details
    last_blur = img_data["last_blur"]
    pre_blur_state = img_data["pre_blur_state"]
    
    # Save current state before modification
    img_data["undo_history"].append(img_data["current_image"].copy())
    
    # Get the pre-blur image state to start from
    image = img_data["current_image"].copy()
    pre_blur_image = pre_blur_state["image"]
    
    # Re-apply the blur with new radius based on the blur type
    if last_blur["type"] == "face":
        # Extract face coordinates
        if "face" in last_blur and last_blur["face"]:
            fx, fy, fw, fh = last_blur["face"]
            
            # Copy the original (unblurred) face region from the pre-blur state
            image[fy:fy+fh, fx:fx+fw] = pre_blur_image[fy:fy+fh, fx:fx+fw]
            
            # Apply new blur
            roi = image[fy:fy+fh, fx:fx+fw]
            blurred = cv2.GaussianBlur(roi, (0, 0), request.radius)
            image[fy:fy+fh, fx:fx+fw] = blurred
    
    elif last_blur["type"] == "custom":
        # Get the mask from pre-blur state
        if "mask" in pre_blur_state:
            mask = pre_blur_state["mask"]
            
            # Create a mask for merging
            mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
            
            # First, restore the original pixels in the masked area
            image = image * (1 - mask_3d) + pre_blur_image * mask_3d
            
            # Now create a newly blurred version of the current image
            blurred = cv2.GaussianBlur(image, (0, 0), request.radius)
            
            # Apply the new blur to just the masked area
            image[:] = image * (1 - mask_3d) + blurred * mask_3d
    
    # Update the last blur operation with new radius
    img_data["last_blur"]["radius"] = request.radius
    
    # Update current image
    img_data["current_image"] = image
    
    # Save result
    result_path = f"results/{image_id}{img_data['file_extension']}"
    cv2.imwrite(result_path, image)
    
    return {"status": "success", "preview_url": f"/images/{image_id}{img_data['file_extension']}"}

@app.post("/undo/{image_id}")
async def undo_operation(image_id: str):
    """Undo the last operation"""
    if image_id not in image_data:
        raise HTTPException(status_code=404, detail="Image not found")
    
    img_data = image_data[image_id]
    
    if not img_data["undo_history"]:
        # No history to undo, return original image
        img_data["current_image"] = img_data["original_image"].copy()
        img_data["pre_blur_state"] = None
        img_data["last_blur"] = None
    else:
        # Restore last state
        img_data["current_image"] = img_data["undo_history"].pop()
        
        # If we've undone the last blur operation, clear the last blur data
        if len(img_data["undo_history"]) == 0:
            img_data["pre_blur_state"] = None
            img_data["last_blur"] = None
    
    # Save result
    result_path = f"results/{image_id}{img_data['file_extension']}"
    cv2.imwrite(result_path, img_data["current_image"])
    
    return {"status": "success", "preview_url": f"/images/{image_id}{img_data['file_extension']}"}

@app.post("/reset/{image_id}")
async def reset_image(image_id: str):
    """Reset image to original state"""
    if image_id not in image_data:
        raise HTTPException(status_code=404, detail="Image not found")
    
    img_data = image_data[image_id]
    img_data["current_image"] = img_data["original_image"].copy()
    img_data["undo_history"] = []
    img_data["pre_blur_state"] = None
    img_data["last_blur"] = None
    
    # Save result
    result_path = f"results/{image_id}{img_data['file_extension']}"
    cv2.imwrite(result_path, img_data["current_image"])
    
    return {"status": "success", "preview_url": f"/images/{image_id}{img_data['file_extension']}"}

@app.post("/save/{image_id}")
async def save_final_image(image_id: str, filename: Optional[str] = Form(None)):
    """Save the processed image with optional filename"""
    if image_id not in image_data:
        raise HTTPException(status_code=404, detail="Image not found")
    
    img_data = image_data[image_id]
    
    # Use original filename if no new name provided
    if not filename:
        name_parts = os.path.splitext(img_data["filename"])
        filename = f"{name_parts[0]}_blurred{name_parts[1]}"
    
    # Ensure filename has the correct extension
    if not filename.endswith(img_data["file_extension"]):
        filename = f"{os.path.splitext(filename)[0]}{img_data['file_extension']}"
    
    # Create a copy for download
    download_path = f"temp/{filename}"
    cv2.imwrite(download_path, img_data["current_image"])
    
    return FileResponse(
        path=download_path, 
        filename=filename,
        media_type=f"image/{img_data['file_extension'][1:]}"
    )

@app.get("/status/{image_id}")
async def get_image_status(image_id: str):
    """Get information about an image"""
    if image_id not in image_data:
        raise HTTPException(status_code=404, detail="Image not found")
    
    img_data = image_data[image_id]
    
    # Get image dimensions
    height, width = img_data["current_image"].shape[:2]
    
    # Convert faces to list of dictionaries
    face_list = []
    for (x, y, w, h) in np.array(img_data["faces"]):
        face_list.append({"x": int(x), "y": int(y), "width": int(w), "height": int(h)})
    
    return {
        "image_id": image_id,
        "original_filename": img_data["filename"],
        "faces_detected": len(img_data["faces"]),
        "faces": face_list,
        "preview_url": f"/images/{image_id}{img_data['file_extension']}",
        "undo_available": len(img_data["undo_history"]) > 0,
        "last_blur_available": img_data["last_blur"] is not None,
        "original_width": width,
        "original_height": height
    }

@app.delete("/cleanup/{image_id}")
async def cleanup_image(image_id: str):
    """Remove image data and files"""
    if image_id not in image_data:
        raise HTTPException(status_code=404, detail="Image not found")
    
    img_data = image_data[image_id]
    
    # Remove files
    try:
        os.remove(img_data["original_path"])
        os.remove(f"results/{image_id}{img_data['file_extension']}")
    except Exception as e:
        logger.error(f"Error cleaning up files: {str(e)}")
    
    # Remove from memory
    del image_data[image_id]
    
    return {"status": "success"}

# Scheduled cleanup task could be added here with background tasks

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=5000, reload=True)
