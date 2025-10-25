import cv2
import numpy as np

def create_mask(cap, BF, JF, JA, BA):
    # Trapéz alakú maszk készül az első frame alapján
    ret, first = cap.read()
    H, W = first.shape[:2]
    mask = np.zeros((H, W), np.uint8)
    points = np.array([
        (int(W* BF[0]), int(H* BF[1])),           # bal felső
        (int(W* JF[0]), int(H* JF[1])),           # jobb felső
        (int(W* JA[0]), int(H* JA[1])),           # jobb alsó
        (int(W* BA[0]), int(H* BA[1]))            # bal alsó 
    ], np.int32)
    cv2.fillPoly(mask, [points], 255)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return mask

def frame_preparation(frame, mask, blur_kernel):
    frame_roi = cv2.bitwise_and(frame, frame, mask=mask)
    gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, blur_kernel, 0)
    return frame_roi, gray_blur

def main(params):
    cap = cv2.VideoCapture(params[0])
    if not cap.isOpened():
        print("Nem sikerült megnyitni a videót")
        exit()
        
    mask = create_mask(cap, params[2], params[3], params[4], params[5])
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_roi, gray_blur = frame_preparation(frame, mask, params[6])
        
        cv2.imshow("Eredeti", frame)
        cv2.imshow("ROI, csak az út", frame_roi)
        cv2.imshow("Gray + Blur", gray_blur)
        if cv2.waitKey(params[1]) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


# A megadható paraméterek
params = [src := "C:/út/a/videóhoz.mp4",
waitkey := 30,
BF := (0.55, 0.00), JF := (0.70, 0.00), JA := (1.00, 1.00), BA :=(0.15, 1.00), # maszk paraméterei
blur_kernel := (5,5)]
main(params)