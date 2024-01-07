import cv2
import pygame
import threading

# Global variables
WIDTH, HEIGHT = 640, 480
VIDEO_FILE_1 = 'video1.mp4'
VIDEO_FILE_2 = 'video2.mp4'

# Thread stop events
stop_event_1 = threading.Event()
stop_event_2 = threading.Event()

def video_thread(filename, screen, stop_event):
    video = cv2.VideoCapture(filename)

    while not stop_event.is_set():
        ret, frame = video.read()

        if not ret:
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset video to the beginning for looping

        # Convert OpenCV frame to Pygame surface
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pygame = pygame.surfarray.make_surface(frame_rgb)
        frame_pygame = pygame.transform.scale(frame_pygame, (WIDTH // 2, HEIGHT))

        # Blit the frame onto the screen
        screen.blit(frame_pygame, (0, 0) if stop_event == stop_event_1 else (WIDTH // 2, 0))

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                stop_event.set()

    video.release()

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Dual Video Player')

# Create threads
thread_1 = threading.Thread(target=video_thread, args=(VIDEO_FILE_1, screen, stop_event_1), daemon=True)
thread_2 = threading.Thread(target=video_thread, args=(VIDEO_FILE_2, screen, stop_event_2), daemon=True)

# Start threads
thread_1.start()
thread_2.start()

# Run Pygame loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

# Set stop events
stop_event_1.set()
stop_event_2.set()

# Wait for threads to finish
thread_1.join()
thread_2.join()

# Clean up Pygame
pygame.quit()
