class PIDController:
    """
    A standard PID controller implementation with anti-windup clamping.
    Used for hardware data collection, baseline control, and autotuning.
    """
    def __init__(self, kp=1.0, ki=0.0, kd=0.0, target=35.0, min_out=0.0, max_out=255.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.target = target
        self.min_out = min_out
        self.max_out = max_out
        
        self.reset()

    def reset(self):
        """Clears controller integral and derivative memory."""
        self.integral = 0.0
        self.prev_error = None

    def set_parameters(self, kp, ki, kd):
        """Dynamically update controller gains."""
        self.kp = kp
        self.ki = ki
        self.kd = kd

    def set_target(self, target):
        """Update the setpoint temperature."""
        self.target = target

    def compute(self, current_temp, dt=1.0):
        """
        Computes the PID control action.
        current_temp: current measured temperature.
        dt: time step in seconds (default is 1.0s).
        Returns:
            clamped control output (float, 0..255)
        """
        if dt <= 0:
            dt = 1.0
            
        error = self.target - current_temp
        
        # Proportional term
        P = self.kp * error
        
        # Integral term with anti-windup clamping
        self.integral += error * dt
        I = self.ki * self.integral
        
        # Derivative term
        if self.prev_error is not None:
            D = self.kd * (error - self.prev_error) / dt
        else:
            D = 0.0
            
        # Raw control signal
        output = P + I + D
        
        # Clamp output and perform conditional integration (anti-windup)
        clamped_output = max(self.min_out, min(self.max_out, output))
        
        # Anti-windup: if output saturated and error is in same direction,
        # freeze/clamp the integral error accumulation to prevent winding up
        if output != clamped_output:
            # If saturated high and accumulating positive error, or saturated low and accumulating negative error
            if (output > self.max_out and error > 0) or (output < self.min_out and error < 0):
                self.integral -= error * dt  # undo integration
                I = self.ki * self.integral
                output = P + I + D
                clamped_output = max(self.min_out, min(self.max_out, output))
                
        self.prev_error = error
        return clamped_output
