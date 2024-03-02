import numpy as np
import os

import pygame
import cv2

pygame.init()

filename = 'bottle_2.avi'
frames_per_second = 24.0
res = '480p'


screen_width = 1000
screen_height = 600

camera_height = 500
camera_width = 350

# Set resolution for the video capture
# Function adapted from https://kirr.co/0l6qmh
def change_res(cap, width, height):
    cap.set(3, width)
    cap.set(4, height)

# Standard Video Dimensions Sizes
STD_DIMENSIONS =  {
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4k": (3840, 2160),
}


# grab resolution dimensions and set video capture to it.
def get_dims(cap, res='1080p'):
    width, height = STD_DIMENSIONS["480p"]
    if res in STD_DIMENSIONS:
        width,height = STD_DIMENSIONS[res]
    ## change the current caputre device
    ## to the resulting resolution
    change_res(cap, width, height)
    return width, height

# Video Encoding, might require additional installs
# Types of Codes: http://www.fourcc.org/codecs.php
VIDEO_TYPE = {
    'avi': cv2.VideoWriter_fourcc(*'XVID'),
    #'mp4': cv2.VideoWriter_fourcc(*'H264'),
    'mp4': cv2.VideoWriter_fourcc(*'XVID'),
}

def get_video_type(filename):
    filename, ext = os.path.splitext(filename)
    if ext in VIDEO_TYPE:
      return  VIDEO_TYPE[ext]
    return VIDEO_TYPE['avi']


screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Camera App")

# Khởi tạo camera
camera = cv2.VideoCapture(1)

out = cv2.VideoWriter(filename, get_video_type(filename), 25, get_dims(camera, res))

# Font
font = pygame.font.Font(None, 24)

# Vòng lặp chính
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            
    screen.fill((192,192,192))
    ret, frame = camera.read()
    
    if ret:
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out.write(frame)
        # Resize frame to show it on UI
        frame = cv2.resize(frame, (camera_height, camera_width))      
        frame = cv2.flip(frame, 1)
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        frame = pygame.surfarray.make_surface(frame)
        screen.blit(frame, (10, 10))


    pygame.display.flip()

camera.release()
pygame.quit()








# cap = cv2.VideoCapture(1)
# out = cv2.VideoWriter(filename, get_video_type(filename), 25, get_dims(cap, res))

# while True:
#     ret, frame = cap.read()
#     out.write(frame)
#     cv2.imshow('frame',frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break


# cap.release()
# out.release()
# cv2.destroyAllWindows()
