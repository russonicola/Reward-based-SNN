from flask import Flask, request, jsonify
from time import sleep, time
import threading
import json
import os
from datetime import datetime
from src.mcu.mcu import MCU, get_os_handler
from shared_memory import SharedMemoryManager


# Load configuration file
with open('config.json', 'r') as f:
    config = json.load(f)
    
VMCU_MODE_HTTP = config['vmcu_mode_http']

NUM_BLOCKS = 5
ROOT_DIR = 'statistics/'

if VMCU_MODE_HTTP:
    CONFIG_FILE = "mem_setup.json"
    manager = SharedMemoryManager(CONFIG_FILE)

# Initialize accuracy counting
correct_guesses = 0
total_balls = 0
balls_per_position = {i: {'total': 0, 'correct': 0, 'predicted': [0 for _ in range(NUM_BLOCKS)], 'lost': 0} for i in range(NUM_BLOCKS)}
current_block_index = None

predicted_position = []
is_valid = False
start_time = None
pred_time = None

device = str(config['device'])
model = 'SNN'+str(config['input_neurons'])
background_color = None
ball_color = None
ball_speed = None
ball_delay = None
trajectories_direction = None
fit_lane = None
distribution = None
servo_delay = None

work_dir = None

experiment_count = 0

servo_delay = config['servo_delay']





    
# Get the current datetime
current_datetime = datetime.now()

# Format the datetime as 'yyyymmdd_hhmmss'
formatted_datetime = current_datetime.strftime('%Y%m%d_%H%M%S')

# create folder
work_dir = ROOT_DIR + formatted_datetime + '_' + device


try:
    os.makedirs(work_dir, exist_ok=True)
except Exception as e:
    print(f"Error creating directory '{work_dir}': {e}")





# Init server app
app = Flask(__name__)


def reset_variables():
    global correct_guesses, current_block_index, total_balls, balls_per_position, predicted_position, is_valid, start_time, pred_time
    global background_color, ball_color, ball_speed, ball_delay, trajectories_direction, fit_lane, distribution

    correct_guesses = 0
    total_balls = 0
    balls_per_position = {i: {'total': 0, 'correct': 0, 'predicted': [0 for _ in range(NUM_BLOCKS)], 'lost': 0} for i in range(NUM_BLOCKS)}
    current_block_index = None
    predicted_position = []
    is_valid = False
    start_time = None
    pred_time = None
    
    background_color = None
    ball_color = None
    ball_speed = None
    ball_delay = None
    trajectories_direction = None
    fit_lane = None
    distribution = None


def get_statistics():
    global correct_guesses, current_block_index, total_balls, balls_per_position
    global device, model, background_color, ball_color, ball_speed, ball_delay, trajectories_direction, fit_lane, distribution
    
    accuracy = correct_guesses / total_balls if total_balls > 0 else 0
    return {
        'device': device,
        'model': model,
        'background_color': background_color,
        'ball_color': ball_color,
        'ball_speed': ball_speed,
        'ball_delay': ball_delay,
        'trajectories_direction': trajectories_direction,
        'fit_lane': fit_lane,
        'distribution': distribution,
        'accuracy': accuracy,
        'correct_guesses': correct_guesses,
        'total_balls': total_balls,
        'balls_per_position': balls_per_position
    }


#▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔#
#                                                                    #
#                        VIRTUAL MCU SERVICES                        #
#                                                                    #
#____________________________________________________________________#

@app.route('/init', methods=['POST'])
def init():
    global background_color, ball_color, ball_speed, ball_delay, trajectories_direction, fit_lane, distribution, work_dir, experiment_count

    experiment_count += 1
    
    reset_variables()

    data = request.json
    
    background_color = data.get('background_color')
    ball_color = data.get('ball_color')
    ball_speed = data.get('ball_speed')
    ball_delay = data.get('ball_delay')
    trajectories_direction = data.get('trajectories_direction')
    fit_lane = data.get('fit_lane')
    distribution = data.get('distribution')
    
    return jsonify({
        'init': True
    })
    


@app.route('/start_ball', methods=['GET'])
def start_ball():
    global predicted_position, is_valid, start_time
    
    start_time = time()

    is_valid = True
    predicted_position = []
    return jsonify({
        'started': True
    })

    
@app.route('/prediction', methods=['POST'])
def prediction():
    global predicted_position, pred_time, servo_delay
    data = request.json
    
    if pred_time is None:
        # First prediction
        pred_time = time()
        predicted_position.append(data.get('position'))
        return jsonify({
            'prediction': True
        })
        
    elif (time() - (pred_time + servo_delay)) > 0:
        # New prediction
        pred_time = time()
        predicted_position.append(data.get('position'))
        return jsonify({
            'prediction': True
        })
    else:
        return jsonify({
            'prediction': False
        })
    
    
@app.route('/end_ball', methods=['POST'])
def end_ball():
    global correct_guesses, current_block_index, total_balls, balls_per_position, predicted_position, is_valid, start_time, pred_time, servo_delay 
    data = request.json
    current_block_index = data.get('position')
    
    if current_block_index in [0, 1, 2, 3, 4, 5, 6, 7]:
    
        if is_valid:
            is_valid = False
            
            if (pred_time is not None) and (time() - pred_time) < servo_delay:
                predicted_position.pop()
            
            if len(predicted_position) == 0:
                balls_per_position[current_block_index]['lost'] += 1 
            else:
                balls_per_position[current_block_index]['predicted'][predicted_position[-1]] += 1
                if predicted_position[-1] == current_block_index:
                    correct_guesses += 1
                    balls_per_position[current_block_index]['correct'] += 1
            
        balls_per_position[current_block_index]['total'] += 1
        total_balls += 1
    
        return jsonify({
            'curr_position': current_block_index
        })
        
    else:
        return jsonify({
            'out_of_target': current_block_index
        })
    

    
@app.route('/reset', methods=['GET'])
def reset():
    reset_variables()
    
    return jsonify({
        'reset': True
    })
    
    
@app.route('/save', methods=['GET'])
def save():
    global work_dir
    
    data = get_statistics()
    
    filename = device+'_'+model+'_'+str(ball_speed)+'_'+background_color[0]+ball_color[0]+'_'+trajectories_direction+'_'+distribution+'_'+str(experiment_count)+'.json'
    
    with open(work_dir+'/'+filename, 'w') as outfile:
        outfile.write(json.dumps(data, indent = 2, ensure_ascii = True))
    
    return jsonify({
        'saved': work_dir+'/'+filename
    })
    
    
@app.route('/accuracy', methods=['GET'])
def accuracy():
    return jsonify(get_statistics())




#▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔#
#                                                                    #
#                       PHISICAL MCU SERVICES                        #
#                                                                    #
#____________________________________________________________________#



mcu = get_os_handler()

def periodic_function():
    mcu.check_timer()
    # Reschedule the function
    threading.Timer(0.5, periodic_function).start()
    
    
def set_servo_fn(angle):
    mcu.timer = time()
    print('angle:',angle)
    mcu.set_servo(angle)

#@app.before_first_request
#def initialize():
#   periodic_function()

@app.route('/set_servo', methods=['POST'])
def set_servo():
    mcu.timer = time()
    angle = request.json.get('angle')
    
    if angle is None or not (-89 <= angle <= 89):
        return jsonify({"error": "Invalid angle"}), 400
    
    set_servo_fn(angle)
    return jsonify({"message": "Servo angle set to {}".format(angle)}), 200

@app.route('/get_ir', methods=['GET'])
def get_ir():
    ir_status = mcu.get_ir()
    return jsonify({"ir_pressed": ir_status})


# sm priorities:
# set servo -> priority 0 writable, priority 1 written. Vmcu sets 1 when 0 and control sets 0 when 1
# get_reward-> priority 0 new value, set and write priority 1. Control sets to 0 when there is a new value, control restore to 0 when read.


if __name__ == '__main__':
    periodic_function()
    if VMCU_MODE_HTTP:
        app.run(host='0.0.0.0', port=5001)
    else:
        try:
            config = manager.load_config()
            
            mem_name = 'shm_servo'
            mem_config = config[mem_name]
            shm_servo, _ = manager.load_memory(mem_name, tuple(mem_config["size"]), mem_config["dtype"])
            
            mem_name = 'shm_reward'
            mem_config = config[mem_name]
            shm_reward, _ = manager.load_memory(mem_name, tuple(mem_config["size"]), mem_config["dtype"])
            
            reward_signal = False # use timer 100ms
            
            while True:
                # read sm
                value, priority = shm_servo.tolist()[0]
                if priority == 0: # and servo is set
                    set_servo_fn(value)
                    shm_servo[:] = [value, 1]
                    
                    
                # Reward
                value, priority = shm_reward.tolist()[0]
                
                if reward_signal is False and mcu.get_ir():
                    reward_signal = True
                    
                if priority == 2 and reward_signal: # and reward is true
                    shm_reward[:] = [1, 0]
                    reward_signal = False
                    
                
            
        finally:
            print("Writer client shutting down.")
        