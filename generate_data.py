import cv2
import numpy as np
import os

RECTS = {
    "top":         (9, 4, 22, 6),
    "upper_right": (20, 4, 22, 15),
    "lower_right": (20, 16, 22, 27),
    "bottom":      (9, 25, 22, 27),
    "lower_left":  (9, 16, 11, 27),
    "upper_left":  (9, 4, 11, 15),
    "middle":      (9, 14, 22, 16),
}

SEGMENTS = {
    0: ["top", "upper_left", "upper_right", "lower_left", "lower_right", "bottom"],
    1: ["upper_right", "lower_right"],
    2: ["top", "upper_right", "middle", "lower_left", "bottom"],
    3: ["top", "upper_right", "middle", "lower_right", "bottom"],
    4: ["upper_left", "middle", "upper_right", "lower_right"],
    5: ["top", "upper_left", "middle", "lower_right", "bottom"],
    6: ["top", "upper_left", "middle", "lower_left", "lower_right", "bottom"],
    7: ["top", "upper_right", "lower_right"],
    8: list(RECTS),
    9: ["top", "upper_left", "upper_right", "middle", "lower_right", "bottom"],
}

def draw_digit(d):
    img = np.zeros((32, 32), np.uint8)
    for s in SEGMENTS[d]:
        x1, y1, x2, y2 = RECTS[s]
        cv2.rectangle(img, (x1, y1), (x2, y2), 255, -1)  # -1 = filled
    return img



def augment(img):
    img = shift(img)
    img = brightness(img)
    img = background(img)
    ing = noise(img)
    
    if np.random.rand() < 0.5:    # rand() gives 0..1, so this is true half the time
        img = blur(img)
        
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img
        
def shift(img):
    dx = np.random.randint(-3, 4)
    dy = np.random.randint(-3, 4)
    M = np.float32([[1, 0, dx],
                    [0, 1, dy]])
    h, w = img.shape
    return cv2.warpAffine(img, M, (w, h))

def brightness(img):
    factor = np.random.uniform(0.5, 1.0)
    return img * factor
    
def background(img):
    return img + np.random.uniform(0, 40)
    
def blur(img):
    return cv2.GaussianBlur(img, (3, 3), 0)
    
def make_number(n_digits=5):
    digits = np.random.randint(0, 10, n_digits)
    img = np.hstack([shift(draw_digit(d)) for d in digits])    
    return augment(img), "".join(map(str, digits))      

def noise(img):
    return img + np.random.normal(0, 10, img.shape)
    
N = 20000

os.makedirs("data/meter_images", exist_ok=True)

with open("data/meter_images/labels.csv", "w") as f:
    f.write("filename,label\n")
    for i in range(N):
        img, label = make_number()
        name = f"{i:05d}.png"
        cv2.imwrite(f"data/meter_images/{name}", img)
        f.write(f"{name},{label}\n")
