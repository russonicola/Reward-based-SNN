import json
from abc import ABC, abstractmethod

device = 'Rpi'
with open('config.json', 'r') as f:
    device = json.load(f)['device']


from time import sleep, time
import asyncio

if device != 'Laptop':
    from gpiozero import Servo, Button
    
IR_PIN = 23
SERVO_PIN = 18

class MCU(ABC):
    @abstractmethod
    def set_servo(self, angle):
        pass
    
    @abstractmethod
    def set_ir(self):
        pass
    
    @abstractmethod
    def get_ir(self):
        pass
    
    @abstractmethod
    def check_timer(self):
        pass
    
    @abstractmethod
    async def check_timer_async(self):
        pass
    
    
class MCU_virtual(MCU):
    def __init__(self):
        pass
    
    def set_servo(self, angle):
        pass
    
    def set_ir(self):
        pass
    
    def get_ir(self):
        pass
    
    def check_timer(self):
        pass
    
    async def check_timer_async(self):
        pass
    

class MCU_physical(MCU):
    def __init__(self):
        self.pressed = False
        self.ir_receiver = Button(IR_PIN, pull_up=True)
        self.ir_receiver.when_pressed = self.set_ir
        self.servo = Servo(SERVO_PIN, 
                     min_pulse_width=0.0009, 
                     max_pulse_width=0.0021)
        self.servo.value=None
        self.position = 0

        self.timer = time()
        self.reset = 2

    def set_servo(self, angle):
        self.timer = time()
        self.servo.value=angle
        self.position = angle
        #sleep(0.1)
        #self.servo.value=None
        #sleep(0.05)

    def set_ir(self):
        self.pressed = True

    def get_ir(self):
        pressed = self.pressed
        self.pressed = False
        return pressed
    
    def check_timer(self):
        if (self.position != 0) and (time() > (self.timer+self.reset)):
            self.set_servo(0)
    
    async def check_timer_async(self):
        while True:
            self.check_timer()
            await asyncio.sleep(0.5)
            
        
def is_raspberry_pi():
    try:
        with open("/proc/cpuinfo", "r") as cpuinfo:
            for line in cpuinfo:
                if "Raspberry Pi" in line or "BCM" in line:
                    return True
        return False
    except FileNotFoundError:
        return False
    
    
def get_os_handler():
    if device == 'Laptop':
        return MCU_virtual()
    elif device == 'Rpi':
        return MCU_physical()
    else:
        raise NotImplementedError(f"Device {device} is not supported")