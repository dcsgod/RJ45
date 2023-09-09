import cv2
import numpy as np

def dark_channel(im, sz):
    b, g, r = cv2.split(im)
    min_channel = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
    return cv2.erode(min_channel, kernel)

def atmospheric_light(im, dark, percentile=0.001):
    h, w = dark.shape
    flat_dark = dark.reshape(h * w)
    num_pixels = int(h * w * percentile)
    indices = np.argpartition(flat_dark, -num_pixels)[-num_pixels:]
    candidates = im.reshape(h * w, 3)[indices]
    return np.max(candidates, axis=0)

def transmission_map(im, A, dark, omega=0.95, t0=0.1):
    h, w = dark.shape
    t = 1 - omega * (dark / A)
    t[t < t0] = t0  # Ensure a minimum value for transmission
    return t

def guided_filter(p, I, radius=40, eps=1e-3):
    mean_p = cv2.boxFilter(p, -1, (radius, radius))
    mean_I = cv2.boxFilter(I, -1, (radius, radius))
    mean_Ip = cv2.boxFilter(I * p, -1, (radius, radius))
    cov_Ip = mean_Ip - mean_I * mean_p
    mean_II = cv2.boxFilter(I * I, -1, (radius, radius))
    var_I = mean_II - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = cv2.boxFilter(a, -1, (radius, radius))
    mean_b = cv2.boxFilter(b, -1, (radius, radius))
    q = mean_a * I + mean_b
    return q

def refine_transmission(t, im, radius=40):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    t = guided_filter(t, gray, radius=radius)
    return t

def recover_scene(im, A, t, t0=0.1):
    h, w, _ = im.shape
    t = np.maximum(t, t0)  # Ensure a minimum value for transmission
    result = np.zeros_like(im, dtype=np.float32)
    for i in range(3):
        result[:, :, i] = ((im[:, :, i].astype(np.float32) - A[i]) / t)
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result

if __name__ == '__main__':
    image_path = "car.jpg"
    input_im = cv2.imread(image_path)
    
    patch_size = 15
    omega = 0.95
    percentile = 0.001
    t0 = 0.1
    guided_radius = 40

    dark = dark_channel(input_im, patch_size)
    A = atmospheric_light(input_im, dark, percentile)
    t = transmission_map(input_im, A, dark, omega, t0)
    refined_t = refine_transmission(t, input_im, guided_radius)
    result = recover_scene(input_im, A, refined_t, t0)
    
    cv2.imwrite('result.jpg', result)
