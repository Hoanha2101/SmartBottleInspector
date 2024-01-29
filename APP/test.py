from ultralytics import YOLO
import cv2
import time

model_track = YOLO("model_set/22.pt")
model_cls = YOLO("model_set/best.pt")

# results_ = model_cls("bus.jpg")

# print(results_[0].names[results_[0].probs.top1])
# frame = results_[0].plot()

# cv2.imshow("f",frame)
# cv2.waitKey()
# cv2.destroyAllWindows()

camera = cv2.VideoCapture("vi.mp4")

run = True

while run:
    
    ret, frame = camera.read()
    start_time = time.time()
    if ret:
        results = model_track.track(frame, persist = True)
        for result in results[0]:
        
            boxes = result.boxes.cpu().numpy() 

            for i in range(len(boxes)):  
                box = boxes[i]
                
                if box.cls[0] == 1:
                    x_min_bottle, y_min_bottle, x_max_bottle, y_max_bottle = map(int,(box.xyxy[0][0], box.xyxy[0][1], box.xyxy[0][2], box.xyxy[0][3]))
                    
                    img = frame[y_min_bottle:y_max_bottle, x_min_bottle:x_max_bottle]
                    
                    results_ = model_cls(img)
                    name = results_[0].names[results_[0].probs.top1]
                    cv2.rectangle(frame, (x_min_bottle, y_min_bottle), (x_max_bottle, y_max_bottle), (0, 255, 0), 2)
                    cv2.putText(frame, str(name), (x_min_bottle, y_min_bottle - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    # Kết thúc thời gian đo FPS
    end_time = time.time()
    
    # Tính FPS
    fps = 1.0 / (end_time - start_time)

    # Hiển thị FPS lên khung hình
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
       
    cv2.imshow("f", frame)        

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên camera và đóng cửa sổ hiển thị
camera.release()
cv2.destroyAllWindows()
        
        