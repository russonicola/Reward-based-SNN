import pygame
import random
import sys
import time
import os
import math
import json
import numpy as np
from shared_memory import SharedMemoryManager

# Configuration
CONFIG_FILE = "mem_setup.json"

manager = SharedMemoryManager(CONFIG_FILE)



# Load configuration file
with open('config.json', 'r') as f:
    config = json.load(f)
ball_delay = config['ball_delay']

# Initialize Pygame
pygame.init()

background_color = 'white'

# Constants


BACKGROUND_COLOR_TOP = (0, 0, 0)  # Black for the top part
BACKGROUND_COLOR_BOTTOM = (255, 255, 255)  # White for the bottom part
BACKGROUND_COLOR_BOTTOM = (0, 0, 0)  # White for the bottom part
BLOCK_COLOR = (255, 255, 255)  # White for the blocks
BLOCK_COLOR_BOTTOM = (0, 0, 0)  # White for the blocks
BALL_COLOR = (255, 255, 255)  # White color for the ball

COLOR = {
    'black': {'top': (0, 0, 0), 'bottom': (0, 0, 0), 'block': (255, 255, 255), 'ball': (255, 255, 255)},
    'white': {'top': (255, 255, 255), 'bottom': (255, 255, 255), 'block': (0, 0, 0), 'ball': (0, 0, 0)}
}

#SCREEN_WIDTH = 800
SCREEN_HEIGHT = 1200
FOV_Y_LIMIT = 800  # The end of the field of view
NUM_BLOCKS = config['downscaling'][1]
SCREEN_WIDTH = NUM_BLOCKS * 100
BLOCK_HEIGHT = 20
BLOCK_WIDTH = SCREEN_WIDTH // NUM_BLOCKS
BALL_RADIUS = 50

METER_IN_PIXELS = 600 # regulate based on the camera FOV
#BALL_SPEED_Y = METER_IN_PIXELS / 120  # 0.5 meter per second, 120 FPS
#BALL_SPEED_Y = METER_IN_PIXELS / 60  # 1 meter per second, 60 FPS
#BALL_SPEED_Y = METER_IN_PIXELS / 30  # 2 meter per second, 30 FPS
#BALL_SPEED_Y = METER_IN_PIXELS / 15  # 4 meter per second, 30 FPS

BALL_SPEED_Y = [(0.5,METER_IN_PIXELS / 120),
                (1,METER_IN_PIXELS / 60),
                (2,METER_IN_PIXELS / 30),
                (4,METER_IN_PIXELS / 15)]
speed_id = 0

TRAJECTORY_TYPE = [{'type':0,'dir':'straight','fit':True,'dist':'uniform'},
                   {'type':1,'dir':'straight','fit':False,'dist':'uniform'},
                   {'type':2,'dir':'random','fit':False,'dist':'random'}]

trajectory_type_id = 0

# Screen setup
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
#screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.FULLSCREEN)
screen.fill(COLOR[background_color]['top'])
pygame.display.flip()
pygame.display.set_caption("Ball Trajectory Simulation")
clock = pygame.time.Clock()

# Get the actual screen size
info = pygame.display.Info()
SCREEN_REAL_WIDTH = info.current_w
SCREEN_REAL_HEIGHT = info.current_h

SHIFT_RIGHT = (SCREEN_REAL_WIDTH - SCREEN_WIDTH) / 2

# Prepare arrays
initial_lanes = NUM_BLOCKS  # can be 8, 16
input_block_width = SCREEN_WIDTH / initial_lanes
x_pos = [(i * input_block_width) + (input_block_width / 2) for i in range(initial_lanes)]

# shuffle is missing
#random.shuffle(x_pos)

#num_balls = 400
num_balls = 160
#num_balls = 80
num_balls = 7
num_balls = 300


def set_dir(type, lane):
    if type == 0:
        return (x_pos[lane] + SHIFT_RIGHT, 0, 0, BALL_SPEED_Y[speed_id][1])
    elif type == 1:
        return (random.randint(BALL_RADIUS, SCREEN_WIDTH - BALL_RADIUS), 0, 0, BALL_SPEED_Y[speed_id][1])
    elif type == 2:
        angle = random.uniform(-math.pi / 6, math.pi / 6)
        return (random.randint(BALL_RADIUS, SCREEN_WIDTH - BALL_RADIUS), BALL_RADIUS, BALL_SPEED_Y[speed_id][1] * math.sin(angle), BALL_SPEED_Y[speed_id][1] * math.cos(angle))
    


def generate_trajectory(start_column, end_column, speed_index):
    """Generate a single trajectory given start column, end column, and speed."""
    speed_y = BALL_SPEED_Y[speed_index][1]
    x_start = (start_column * BLOCK_WIDTH) + (BLOCK_WIDTH / 2)
    x_end = (end_column * BLOCK_WIDTH) + (BLOCK_WIDTH / 2)
    time_to_reach_fov = FOV_Y_LIMIT / speed_y
    speed_x = (x_end - x_start) / time_to_reach_fov
    return x_start, 0, speed_x, speed_y, x_end


def predict_final_block(start_column, end_column, speed_index):
    """Predict the final block that the ball will reach at the end of the FOV."""
    speed_y = BALL_SPEED_Y[speed_index][1]
    x_start = (start_column * BLOCK_WIDTH) + (BLOCK_WIDTH / 2)
    x_end = (end_column * BLOCK_WIDTH) + (BLOCK_WIDTH / 2)
    time_to_reach_fov = FOV_Y_LIMIT / speed_y
    speed_x = (x_end - x_start) / time_to_reach_fov
    x_final = x_start + speed_x * time_to_reach_fov
    block_final = int(x_final // BLOCK_WIDTH)
    block_final = max(0, min(NUM_BLOCKS - 1, block_final))
    return block_final




def calibrate():
    global background_color
    screen.fill(COLOR[background_color]['top'])
    for rep in range(1001):
        # Draw the white rectangle once
        rect_width = SCREEN_WIDTH - BALL_RADIUS
        rect_height = SCREEN_WIDTH - BALL_RADIUS
        rect_x = SHIFT_RIGHT + (SCREEN_WIDTH - rect_width) // 2
        rect_y = (SCREEN_WIDTH - rect_height) // 2
        pygame.draw.rect(screen, COLOR[background_color]['block'], (rect_x, rect_y, rect_width, rect_height))

        # Draw the "+" symbol in the middle of the rectangle
        if rep % 2 == 0:
            plus_color = (0, 0, 0)  # Black color for the "+"
        else:
            plus_color = (255, 255, 255)  # White color for the "+"
            
        plus_size = BALL_RADIUS  # Size of the "+"
        plus_x_center = rect_x + rect_width // 2
        plus_y_center = rect_y + rect_height // 2
        pygame.draw.line(screen, plus_color, 
                        (plus_x_center - plus_size // 2, plus_y_center), 
                        (plus_x_center + plus_size // 2, plus_y_center), 5)
        pygame.draw.line(screen, plus_color, 
                        (plus_x_center, plus_y_center - plus_size // 2), 
                        (plus_x_center, plus_y_center + plus_size // 2), 5)
        pygame.draw.circle(screen, plus_color, (SHIFT_RIGHT + BALL_RADIUS, BALL_RADIUS), BALL_RADIUS)
        pygame.draw.circle(screen, plus_color, (SHIFT_RIGHT + BALL_RADIUS, SCREEN_WIDTH - BALL_RADIUS), BALL_RADIUS)
        pygame.draw.circle(screen, plus_color, (SHIFT_RIGHT + SCREEN_WIDTH - BALL_RADIUS, BALL_RADIUS), BALL_RADIUS)
        pygame.draw.circle(screen, plus_color, (SHIFT_RIGHT + SCREEN_WIDTH - BALL_RADIUS, SCREEN_WIDTH - BALL_RADIUS), BALL_RADIUS)

        pygame.display.flip()
        time.sleep(0.02)
        
    
# Ball class
class Ball:
    def __init__(self, trajectory):
        self.x, self.y, self.speed_x, self.speed_y, self.x_end = trajectory
        self.radius = BALL_RADIUS
        self.color = COLOR[background_color]['ball']

    def move(self):
        self.x += self.speed_x
        self.y += self.speed_y

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)
        
        
        
        
def ball_loop(shm_array, start_column, end_column, speed_index):
    global background_color, url_type, num_balls
    
    background_color_id = 0 if background_color == 'black' else 1
    
    total_balls = 0

    ball = Ball(generate_trajectory(start_column, end_column, speed_index))

    is_moving = False
    hitted = False
    running = True
    
    while running:
        
        #if total_balls < num_balls:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Fill the screen with the top and bottom background colors
        screen.fill(COLOR[background_color]['top'])
        
        # Move and draw the ball
        ball.move()
        ball.draw(screen)
        
        pygame.draw.rect(screen, COLOR[background_color]['bottom'], (SHIFT_RIGHT, SCREEN_WIDTH, SCREEN_WIDTH, SCREEN_HEIGHT - SCREEN_WIDTH))
        
        # Draw the black background for the blocks
        pygame.draw.rect(screen, COLOR[background_color]['bottom'], (SHIFT_RIGHT, 1180, SCREEN_WIDTH, BLOCK_HEIGHT))
        
        # Draw the white blocks on top of the black background
        for i in range(NUM_BLOCKS):
            pygame.draw.rect(screen, COLOR[background_color]['bottom'], ((i * BLOCK_WIDTH) + SHIFT_RIGHT, 1180, BLOCK_WIDTH, BLOCK_HEIGHT))
        
        if not is_moving:
            is_moving = True
            hitted = False
            # CALL START
            
            # -----------------–-
            # --- PLACEHOLDER ---
            # /start_ball
            # ------------------–
            shm_array[:] = [1, background_color_id, speed_id, trajectory_type_id, -100]

        # Check if the ball has reached the bottom section or gone out of view
        if ball.y - ball.radius > SCREEN_HEIGHT:
            ball = Ball(generate_trajectory(start_column, end_column, speed_index))

        # Check if the ball touches any block in the grid
        if ball.y + ball.radius >= 1180:
            running = False
            
            block_index = int((ball.x - SHIFT_RIGHT) // BLOCK_WIDTH)
            if not hitted: # error here
                hitted = True
                print(f"Ball hit block at coordinate: {int(ball.x)}")
                
                # CALL END
                # -----------------–-
                # --- PLACEHOLDER ---
                # /end_ball -> {'position': block_index}
                # ------------------–
                time.sleep(1)
                shm_array[:] = [2, background_color_id, speed_id, trajectory_type_id, int(ball.x)]
            
            # ball delay
            time.sleep(ball_delay)
            is_moving = False
            total_balls += 1
            ball = Ball(generate_trajectory(start_column, end_column, speed_index))

        pygame.display.flip()
        clock.tick(60)

    time.sleep(2)
    
    # CALL SAVE
    # -----------------–-
    # --- PLACEHOLDER ---
    # /save
    # ------------------–
    #shm_array[:] = [3, -1]
                
# Main game loop
def main(shm_array):
    op = ''
    color_op = ''
    speed_op = ''
    test_op = ''
    
    global background_color, speed_id, trajectory_type_id
    
    while op != '0':
        print("""
            Select operation:
            
            1 - Calibrate DVS Camera
            2 - Start moving balls
            3 - Set background color
            4 - Complete test
            5 - Single test
            0 - Exit
            """)
    
        op = input()
        
        if op == '1':
            calibrate()
        elif op == '2':
            print('Select background color: 0 black, 1 white')
            color_op = input()
            if color_op == '0':
                background_color = 'black'
                print('Background color selected: black')
            elif color_op == '1':
                background_color = 'white'
                print('Background color selected: white')
            else:
                print('Wrong selection')
                continue
            
            screen.fill(COLOR[background_color]['top'])
            pygame.display.flip()
            
            print('Select 0-3 for speed 0.5m/s, 1m/s, 2m/s, 4m/s')
            speed_op = input()
            if speed_op == '0':
                speed_id = 0
                print('Start test at 0.5m/s')
            elif speed_op == '1':
                speed_id = 1
                print('Start test at 1m/s')
            elif speed_op == '2':
                speed_id = 2
                print('Start test at 2m/s')
            elif speed_op == '3':
                speed_id = 3
                print('Start test at 04m/s')
            else:
                speed_id = 0
                print('Wrong selection')
                continue
            
            print('Select 0-2 for straight, in lane random test, straight random test, all dir random test')
            test_op = input()
            if test_op == '0':
                trajectory_type_id = 0
                print('Start in lane random test')
            elif test_op == '1':
                trajectory_type_id = 1
                print('straight random test')
            elif test_op == '2':
                trajectory_type_id = 2
                print('Start all dir random test')
            else:
                trajectory_type_id = 0
                print('Wrong selection')
                continue
            
            time.sleep(1)
            if speed_op in ['0','1','2','3'] and test_op in ['0','1','2']:
                match trajectory_type_id:
                    case 0:
                        for column_start in range(8):
                            ball_loop(shm_array, column_start, column_start, speed_id)
                    case 1:
                        for column_start in range(8):
                            for column_end in range(8):
                                ball_loop(shm_array, column_start, column_end, speed_id)
                        
        elif op == '3':
            
            print('Select 1 for black, 2 for white')
            col = input()
            if col == '1':
                background_color = 'black'
                screen.fill(COLOR[background_color]['top'])
                pygame.display.flip()
                print('black color background selected')
            elif col == '2':
                background_color = 'white'
                screen.fill(COLOR[background_color]['top'])
                pygame.display.flip()
                print('white color background selected')
            else:
                print('wrong selection')
                
        elif op == '4':
            for color in ['white']: # ['black','white']
                for speed in range(4): # 4
                    for column_start in range(8):
                        for column_end in range(8):
                            background_color = color
                            speed_id = speed
                            ball_loop(shm_array, column_start, column_end, speed) # pass shared memory
                            
        elif op == '5':
            start_column = int(input(f"Enter start column (0 to {NUM_BLOCKS - 1}): "))
            end_column = int(input(f"Enter end column (0 to {NUM_BLOCKS - 1}): "))
            speed_index = int(input("Select speed (0: 0.5m/s, 1: 1m/s, 2: 2m/s, 3: 4m/s): "))
            predicted_block = predict_final_block(start_column, end_column, speed_index)
            print(f"Predicted final block: {predicted_block}")
            ball_loop(shm_array, start_column, end_column, speed_index) # pass shared memory
                
        elif op == '0':
            print("Bye")
        else:
            print("Incorrect selection. \n\n")
            

if __name__ == "__main__":
    try:
        # Load shared memory for writing
        config = manager.load_config()
        #mem_arr = [mem for mem in config["shared_memories"] if mem["role"] in ["writer", "shared"]]
        #mem_config = next(mem for mem in config["shared_memories"] if mem["role"] in ["writer"])
        mem_name = 'shm_record'
        mem_config = config[mem_name]
        shm_array, _ = manager.load_memory(mem_name, tuple(mem_config["size"]), mem_config["dtype"])
        
        main(shm_array)
        
    finally:
        print("Writer client shutting down.")