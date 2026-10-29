import cv2
import numpy as np

cap = cv2.VideoCapture(r"C:\Foundry\Griptape\video_gen_server\outputs\minimax_1792544088_3728417020.mp4")
ret, frame = cap.read()
if ret:
    print("Frame shape:", frame.shape)
    print("Frame mean:", frame.mean(), "std:", frame.std(), "min:", frame.min(), "max:", frame.max())
    cv2.imwrite(r"C:\Foundry\Griptape\video_gen_server\outputs\latest_frame.png", frame)
    print("Saved latest_frame.png")
cap.release()
