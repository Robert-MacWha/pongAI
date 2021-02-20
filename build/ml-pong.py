import pygame
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

# static variables
size = (256, 192)
frameRate = 1
initial_state = np.load('./initial_state.npy')
state_buffer = initial_state

engine        = keras.models.load_model('./engine.h5')
renderer      = keras.models.load_model('./renderer.h5')

# initialize pygame
pygame.init()
display = pygame.display.set_mode(size)
pygame.display.set_caption("PongAI")

clock = pygame.time.Clock()

# main loop
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # detect inputs
    action = [0, 0, 0]

    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        action[0] = 1
    elif keys[pygame.K_DOWN]:
        action[1] = 1
    else:
        action[2] = 1  

    print(action)

    # pass the action & the previous states to the models
    engine_input = np.asarray(state_buffer[1].tolist() + state_buffer[0].tolist() + action).reshape(1, 131)
    new_state = engine.predict(engine_input)
    render = renderer.predict(state_buffer[1].reshape(1, 64))

    # update the state buffer
    state_buffer[1] = state_buffer[0]
    state_buffer[0] = new_state

    # render a surface with the render on it
    render = np.rint( render ).astype("uint8").reshape(256, 192) * 255
    surf = pygame.surfarray.make_surface(render)
    display.blit(surf, (0, 0))

    # update the screen & the clock
    pygame.display.flip()
    clock.tick(frameRate)

pygame.quit()
