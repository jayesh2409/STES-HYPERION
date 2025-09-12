from ultralytics import YOLO
import cv2

model = YOLO(r"D:\autonomous\2025 cone detection\yolov8n_.pt")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot access the camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    results = model(frame)

    annotated_frame = frame.copy()
    for result in results:
        boxes = result.boxes 
        for box in boxes:
            cls_id = int(box.cls.cpu().numpy().item())
            conf = float(box.conf.cpu().numpy().item())
            xyxy = box.xyxy.cpu().numpy()[0]

            class_name = model.names[cls_id]

            x1, y1, x2, y2 = map(int, xyxy)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name} {conf:.2f}"
            cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Cone Detection', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
