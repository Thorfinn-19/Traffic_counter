import cv2

def main(params):
    cap = cv2.VideoCapture(params[0])
    if not cap.isOpened():
        print("Nem sikerült megnyitni a videót!")
        exit()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Eredeti", frame)
        if cv2.waitKey(params[1]) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


params = [src := "C:/út/a/videóhoz.mp4",
        waitkey := 30]
main(params)