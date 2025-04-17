# webcam.py
import cv2 # type: ignore

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("カメラを開けませんでした")
        return

    print("カメラを起動中... qで終了")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
