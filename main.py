from http.client import HTTPException
from fastapi import FastAPI, File, UploadFile, Request, Query
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
import os
import re
import cv2
from moviepy.editor import VideoFileClip
import dlib
from model import align_and_crop_face, process_and_create_video

app = FastAPI()

# Đường dẫn để lưu video upload và video đã xử lý
UPLOAD_DIRECTORY = "F:/PyCharmProjects/FastAPI/uploads"
OUTPUT_DIRECTORY = "outputs"
PROCESSED_DIRECTORY = 'processed'
# Tạo thư mục nếu chưa tồn tại
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

templates = Jinja2Templates(directory="templates")


def sanitize_filename(filename):
    """Làm sạch tên tệp để loại bỏ các ký tự không hợp lệ."""
    return re.sub(r'[<>:"/\\|?*]', '', filename)


@app.post('/uploadvideo/')
async def upload_video(file: UploadFile = File(...)):
    sanitized_filename = sanitize_filename(file.filename)
    file_location = os.path.join(UPLOAD_DIRECTORY, sanitized_filename)
    try:
        # Ghi tệp vào thư mục uploads
        with open(file_location, "wb+") as file_object:
            file_object.write(await file.read())
        # Kiểm tra xem tệp đã được lưu chưa
        if os.path.exists(file_location):
            return {"filename": sanitized_filename}
        else:
            return {"error": "File upload failed!"}
    except Exception as e:
        return {"error": str(e)}


@app.get("/", response_class=HTMLResponse)
async def main(request: Request):
    videos = os.listdir(UPLOAD_DIRECTORY)  # Lấy danh sách video đã upload
    return templates.TemplateResponse("index.html", {"request": request, "videos": videos})


@app.get("/uploads/{filename}")
async def get_video(filename: str):
    file_path = os.path.join(UPLOAD_DIRECTORY, filename)
    print(file_path)
    if os.path.exists(file_path):
        return FileResponse(file_path)  # Trả về tệp video
    return {"error": "File not found"}



from pydantic import BaseModel

class VideoFilename(BaseModel):
    filename: str

def convert_video(input_path: str, output_path: str):
    # Sử dụng moviepy để chuyển đổi định dạng video
    video_clip = VideoFileClip(input_path)
    video_clip.write_videofile(output_path, codec='libx264')

import dlib

# Khởi tạo model phát hiện khuôn mặt
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

@app.post("/processvideo/")
async def process_video(data: VideoFilename):
    filename = data.filename
    if not filename:
        raise HTTPException(status_code=422, detail="Filename is required")
    input_path = os.path.join(UPLOAD_DIRECTORY, filename)
    if not os.path.exists(input_path):
        raise HTTPException(status_code=404, detail="File not found")

    # Đường dẫn cho video sau khi được xử lý
    processed_path = os.path.join(PROCESSED_DIRECTORY, f'processed_{filename}')
    process_and_create_video(input_path, processed_path)


    # Thực hiện chuyển đổi nếu cần
    output_path = os.path.join(PROCESSED_DIRECTORY, f'converted_{filename}')
    convert_video(processed_path, output_path)

    return {
        "message": f"File '{filename}' processed and converted successfully",
        "processed_video": f"processed/converted_{filename}"
    }


@app.get("/processed/{filename}")
async def get_processed_video(filename: str):
    file_path = os.path.join(PROCESSED_DIRECTORY, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)  # Trả về tệp video
    return {"error": "Processed video not found"}
# if __name__ == "__main__":
#     import uvicorn
#
#     uvicorn.run("main:app", host="0.0.0.0", port=8001)
