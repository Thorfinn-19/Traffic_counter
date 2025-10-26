import cv2
import numpy as np
import math

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

# Jármű helyzete. Pozitív: vonal alatt. Negatív: vonal felett
def vehicle_position(A, B, P):
    (Ax, Ay), (Bx, By) = A, B
    Px, Py = P
    side = (Px - Ax)*(By - Ay) - (Py - Ay)*(Bx - Ax)
    return 1 if side > 0 else -1

def crossing_over(centers, prev_pts, prev_pos, count_up, count_down, A, B, MAX_DIST):
    matched_prev = set()    # Már párosított pontok
    for (cx, cy) in centers:
        pos_now = vehicle_position(A, B, (cx, cy)) # Jármű melyik oldalon van most
        # egyszerű nearest-neighbor párosítás az előző frame-hez és átlépés logika
        nearest_neighbor, best_distance = -1, MAX_DIST + 1
        for i, (px, py) in enumerate(prev_pts):
            if i in matched_prev:
                continue
            d = math.hypot(cx - px, cy - py)
            if d < best_distance:
                best_distance, nearest_neighbor = d, i    # Legkisebb távolságú pont
        #átlépés detektálás
        if nearest_neighbor != -1 and best_distance <= MAX_DIST:
            pos_prev = prev_pos[nearest_neighbor] if nearest_neighbor < len(prev_pos) else 0
            if pos_prev * pos_now < 0:
                if pos_now > 0:
                    count_up += 1
                else:
                    count_down += 1
            matched_prev.add(nearest_neighbor)    
    return count_up, count_down

def main(params):
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print("Nem sikerült megnyitni a videót")
        exit()
        
    mask, H, W = create_mask(cap, BF, JF, JA, BA)

    A, B = line_position(y_line, H, W)

    bg, kernel_open, kernel_bridge = foreground_preprocessing(background_history, varThreshold, ellipse_kernel, rect_kernel)

    # Frame to frame tracking a vonalátlépéshez
    prev_pts = []
    prev_pos = []

    # Counters
    count_up = 0
    count_down = 0
    
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_roi, gray_blur = frame_preparation(frame, mask, blur_kernel)

        fg, fg_clean = moving_object_highlighting(gray_blur, bg, kernel_open, kernel_bridge, fg_clean_iter)

        detections, centers = drawings(fg_clean, frame_roi, A, B, MIN_AREA, ASPECT_MIN, ASPECT_MAX)

        count_up, count_down = crossing_over(centers, prev_pts, prev_pos, count_up, count_down, A, B, MAX_DIST)
        # számlálók kiírása
        cv2.putText(detections, f"UP: {count_up}  DOWN: {count_down}  TOTAL: {count_up+count_down}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        
        cv2.imshow("Eredeti", frame)
        cv2.imshow("ROI, csak az út", frame_roi)
        cv2.imshow("Gray + Blur", gray_blur)
        cv2.imshow("FG Mask (MOG2)", fg)
        cv2.imshow("FG (open + dilate)", fg_clean)
        cv2.imshow("Detections (contours)", detections)

        # előző állapot frissítése
        prev_pts = centers
        prev_pos = [vehicle_position(A, B, p) for p in centers]
        
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
y_line := 0.6,
MAX_DIST := 40]

main(params)