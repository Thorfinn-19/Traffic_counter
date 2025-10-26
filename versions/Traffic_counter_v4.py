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
    return mask, H, W

def foreground_preprocessing(background_history=300, varThreshold=25, ellipse_kernel=(5, 5), rect_kernel=(5, 5)):
    bg = cv2.createBackgroundSubtractorMOG2(background_history, varThreshold, detectShadows=True)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ellipse_kernel)
    kernel_bridge = cv2.getStructuringElement(cv2.MORPH_RECT,   rect_kernel)
    return bg, kernel_open, kernel_bridge

def frame_preparation(frame, mask, blur_kernel):
    frame_roi = cv2.bitwise_and(frame, frame, mask=mask)
    gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, blur_kernel, 0)
    return frame_roi, gray_blur

def moving_object_highlighting(gray_blur, bg, kernel_open, kernel_bridge, fg_clean_iter=1):
    fg = bg.apply(gray_blur)  # 0=háttér, 127=árnyék, 255=előtér
    fg[fg == 127] = 0 #árnyék eldobása
    fg_open = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel_open, iterations=1) # Morfológia: OPEN (erode -> dilate)
    fg_close = cv2.morphologyEx(fg_open, cv2.MORPH_CLOSE, kernel_bridge, iterations=20) # Morfológia: OPEN (dilate -> erode)
    fg_clean = cv2.dilate(fg_close, kernel_bridge, iterations=fg_clean_iter)
    return fg, fg_clean

def line_position(y_line, H, W):
    y_line = int(H * y_line)
    A = (0, y_line)
    B = (W, y_line)
    return A, B

def drawings(fg_clean, frame_roi, A, B, MIN_AREA=500, ASPECT_MIN=0.3, ASPECT_MAX=4):
    contours, _ = cv2.findContours(fg_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = frame_roi.copy()
    cv2.line(detections, A, B, (255, 0, 0), 2)
    centers = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_AREA:
            continue
        x, y, w, h = cv2.boundingRect(c)
        aspect = w / float(h) if h > 0 else 0.0
        if aspect < ASPECT_MIN or aspect > ASPECT_MAX:
            continue
        M = cv2.moments(c)
        if M["m00"] <= 0: # m00 = az alakzat mérete
            continue
        cx = int(M["m10"] / M["m00"]) # az alakzat súlypontjának x koordinátája
        cy = int(M["m01"] / M["m00"]) # az alakzat súlypontjának y koordinátája
        centers.append((cx, cy))
        cv2.circle(detections, (cx, cy), 4, (0, 0, 255), -1)
        cv2.rectangle(detections, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return detections, centers

def main(params):
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print("Nem sikerült megnyitni a videót")
        exit()
        
    mask, H, W = create_mask(cap, BF, JF, JA, BA)

    A, B = line_position(y_line, H, W)

    bg, kernel_open, kernel_bridge = foreground_preprocessing(background_history, varThreshold, ellipse_kernel, rect_kernel)
    
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_roi, gray_blur = frame_preparation(frame, mask, blur_kernel)

        fg, fg_clean = moving_object_highlighting(gray_blur, bg, kernel_open, kernel_bridge, fg_clean_iter)

        detections, _ = drawings(fg_clean, frame_roi, A, B, MIN_AREA, ASPECT_MIN, ASPECT_MAX)
        
        cv2.imshow("Eredeti", frame)
        cv2.imshow("ROI, csak az út", frame_roi)
        cv2.imshow("Gray + Blur", gray_blur)
        cv2.imshow("FG Mask (MOG2)", fg)
        cv2.imshow("FG (open + dilate)", fg_clean)
        cv2.imshow("Detections (contours)", detections)
        
        if cv2.waitKey(waitkey) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

    
# A megadható paraméterek
params = [src := "C:/út/a/videóhoz.mp4", # videó forrása
waitkey := 30,
BF := (0.55, 0.00), JF := (0.70, 0.00), JA := (1.00, 1.00), BA :=(0.15, 1.00), # maszk paraméterei,
blur_kernel := (5, 5),
background_history := 300,
varThreshold := 25,
ellipse_kernel := (5, 5),
rect_kernel := (3, 5),
fg_clean_iter := 1,
MIN_AREA := 6000,
ASPECT_MIN := 0.3,
ASPECT_MAX := 4,
y_line := 0.7]

main(params)