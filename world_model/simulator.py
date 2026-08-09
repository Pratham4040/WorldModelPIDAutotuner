import random
import numpy as np

class ThermalSimulator:
    """
    A high-fidelity thermodynamic simulator for the thermal chamber.
    Uses a 2-stage heat transfer model (Heater -> Sensor) to simulate thermal lag,
    ambient dissipation, sensor noise, and environmental drafts.
    """
    def __init__(self, 
                 ambient_temp=25.0, 
                 c_heat=0.08,           # Heater power coefficient
                 c_cool_heater=0.02,     # Heat transfer from heater to sensor
                 c_loss_heater=0.01,     # Heat loss from heater to ambient
                 c_sensor_gain=0.015,    # Heat absorption rate of sensor
                 c_loss_sensor=0.005,    # Heat loss from sensor to ambient
                 sensor_noise_std=0.02,  # Noise in temperature reading
                 safety_temp_limit=39.0  # Hardware safety cutoff
                ):
        self.ambient_temp = ambient_temp
        self.c_heat = c_heat
        self.c_cool_heater = c_cool_heater
        self.c_loss_heater = c_loss_heater
        self.c_sensor_gain = c_sensor_gain
        self.c_loss_sensor = c_loss_sensor
        self.sensor_noise_std = sensor_noise_std
        self.safety_temp_limit = safety_temp_limit
        
        self.reset()

    def reset(self, initial_temp=None):
        """Resets the simulator states."""
        if initial_temp is None:
            initial_temp = self.ambient_temp
            
        self.t_heater = initial_temp
        self.t_sensor = initial_temp
        self.draft_offset = 0.0
        self.steps = 0
        return self.read_temp()

    def step(self, pwm):
        """
        Runs the thermodynamics simulation for 1 second.
        pwm: heater duty cycle (0 to 255)
        """
        self.steps += 1
        
        # Enforce PWM safety limits (0 to 255)
        pwm = max(0, min(255, int(pwm)))
        
        # Physical/Hardware safety cutoff
        if self.t_sensor >= self.safety_temp_limit:
            pwm = 0
            
        # Convert PWM to fractional heating input (0.0 to 1.0)
        u = pwm / 255.0
        
        # Simulate ambient temperature drafts (random walk/slow wave)
        # Draft fluctuates by up to +/- 1.0 deg C over time
        self.draft_offset += random.gauss(0, 0.02)
        self.draft_offset = max(-1.0, min(1.0, self.draft_offset))
        current_ambient = self.ambient_temp + self.draft_offset
        
        # Thermodynamic differential equations (Euler integration, dt=1s)
        # Heater temperature change
        dt_heater = (
            self.c_heat * u * 5.0  # Heater heats up (scaled)
            - self.c_cool_heater * (self.t_heater - self.t_sensor)
            - self.c_loss_heater * (self.t_heater - current_ambient)
        )
        
        # Sensor temperature change (thermal lag / absorption)
        dt_sensor = (
            self.c_sensor_gain * (self.t_heater - self.t_sensor)
            - self.c_loss_sensor * (self.t_sensor - current_ambient)
        )
        
        # Update states
        self.t_heater += dt_heater
        self.t_sensor += dt_sensor
        
        # Prevent physical impossibilities
        self.t_heater = max(0.0, self.t_heater)
        self.t_sensor = max(0.0, self.t_sensor)
        
        return self.read_temp(), pwm

    def read_temp(self):
        """Reads the sensor temperature, adding realistic sensor measurement noise."""
        noise = random.gauss(0, self.sensor_noise_std)
        measured_temp = self.t_sensor + noise
        
        # Safety cutoff check
        if measured_temp >= self.safety_temp_limit:
            return measured_temp, True
        return measured_temp, False
