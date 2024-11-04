import cv2
import dlib
import os
import numpy as np

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def align_and_crop_face(image, face):
    # Dự đoán vị trí các landmark trên khuôn mặt
    shape = predictor(image, face)

    # Xác định mắt trái và mắt phải
    left_eye = (shape.part(36).x, shape.part(36).y)
    right_eye = (shape.part(45).x, shape.part(45).y)

    # Tính toán góc xoay dựa trên vị trí hai mắt
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))

    # Xác định trung điểm giữa hai mắt
    eyes_center = (float(left_eye[0] + right_eye[0]) / 2,
                   float(left_eye[1] + right_eye[1]) / 2)

    # Tạo ma trận xoay dựa trên góc và tâm xoay là mắt
    M = cv2.getRotationMatrix2D(eyes_center, angle, scale=1)
    aligned_face = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    # Crop phần khuôn mặt dựa trên bounding box của khuôn mặt đã phát hiện
    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    cropped_face = aligned_face[y:y + h, x:x + w]

    return cropped_face


# Sử dụng hàm align_and_crop_face
def detect_faces_in_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Không thể đọc hình ảnh")
        return None

    # Xoay ảnh tối đa 3 lần
    for rotation_count in range(4):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        if faces:
            for face in faces:
                cropped_face = align_and_crop_face(image, face)

                # Hiển thị khuôn mặt đã được crop
                cv2.imshow("Aligned and Cropped Face", cropped_face)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                return

        # Xoay ảnh 90 độ nếu chưa phát hiện khuôn mặt
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)


# Tạo thư mục processed nếu chưa tồn tại
PROCESSED_DIRECTORY = "processed"
if not os.path.exists(PROCESSED_DIRECTORY):
    os.makedirs(PROCESSED_DIRECTORY)

def process_and_create_video(video_path, output_video_path, fps=30):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Không thể mở video")
        return

    frame_count = 0  # Đếm số khung hình để tạo tên tệp

    # Lấy thông số khung hình từ video gốc để tạo video mới
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Tạo đối tượng VideoWriter để xuất video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Có thể thay đổi codec nếu cần
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Xoay ảnh tối đa 3 lần
        for rotation_count in range(4):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            if faces:
                for face in faces:
                    cropped_face = align_and_crop_face(frame, face)

                    # Kiểm tra xem cropped_face có rỗng hay không trước khi thêm vào video
                    if cropped_face.size > 0:
                        # Chỉ lấy phần khuôn mặt để đưa vào video
                        face_height, face_width = cropped_face.shape[:2]
                        if face_height != height or face_width != width:
                            padded_face = cv2.resize(cropped_face, (width, height))
                        else:
                            padded_face = cropped_face
                        video_writer.write(padded_face)
                        print(f"Đã thêm khung hình vào video: {frame_count}")

                    frame_count += 1  # Tăng số khung hình đã xử lý
                    break  # Dừng vòng lặp xoay nếu đã phát hiện khuôn mặt

            # Nếu chưa phát hiện khuôn mặt, xoay video 90 độ
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    print(f"Đã lưu video khuôn mặt vào: {output_video_path}")


# Sử dụng hàm để tạo video
if __name__ == '__main__':
    process_and_create_video('VID_20241001_144457.mp4', 'output_video.mp4')