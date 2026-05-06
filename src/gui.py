import pygame
import numpy as np
from PIL import Image
import main
import matplotlib.pyplot as plt

# ===== SETTINGS =====
WINDOW_SIZE = 280
IMG_SIZE = 28
DRAW_SIZE = 20
BRUSH_RADIUS = 12

pygame.init()
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption("Draw Digit (SPACE=Predict, C=Clear)")
screen.fill((0, 0, 0))

clock = pygame.time.Clock()
drawing = False

# ===== YOUR MODEL IMPORT HERE =====
# from your_model_file import predict, model, dp


def preprocess(surface):
    # 1. Get pixel data
    data = pygame.surfarray.array3d(surface)[:, :, 0]

    # 2. Normalize
    data = data / 255.0

    # 3. Threshold (remove noise)
    data = (data > 0.2).astype(float)

    # 4. Find bounding box
    coords = np.argwhere(data > 0)
    if len(coords) == 0:
        return None  # nothing drawn

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    digit = data[y_min:y_max+1, x_min:x_max+1]

    # 5. Resize to 20x20 (MNIST style)
    digit_img = Image.fromarray((digit * 255).astype(np.uint8))
    digit_img = digit_img.resize((DRAW_SIZE, DRAW_SIZE), Image.BILINEAR)

    digit = np.array(digit_img) / 255.0

    # 6. Place into 28x28 center
    new_img = np.zeros((IMG_SIZE, IMG_SIZE))

    x_offset = (IMG_SIZE - DRAW_SIZE) // 2
    y_offset = (IMG_SIZE - DRAW_SIZE) // 2

    new_img[y_offset:y_offset+DRAW_SIZE, x_offset:x_offset+DRAW_SIZE] = digit

    # 7. Optional: invert if needed
    # new_img = 1 - new_img

    return new_img


def draw_loop():
    global drawing
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                drawing = True

            elif event.type == pygame.MOUSEBUTTONUP:
                drawing = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:
                    screen.fill((0, 0, 0))

                elif event.key == pygame.K_SPACE:
                    processed = preprocess(screen)

                    if processed is None:
                        print("Draw something first!")
                        continue

                    flat = processed.flatten()

                    # ===== PREDICTION =====
                    try:
                        prediction = main.predict(flat, main.model, main.dp)
                        print("Prediction:", prediction)
                    except:
                        print("Prediction function not connected yet")

                    # ===== DEBUG VISUAL =====
                    plt.imshow(processed, cmap='gray')
                    plt.title("What the model sees")
                    plt.show()

        if drawing:
            x, y = pygame.mouse.get_pos()
            pygame.draw.circle(screen, (255, 255, 255), (x, y), BRUSH_RADIUS)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


draw_loop()