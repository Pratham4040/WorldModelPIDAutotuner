import numpy as np
import torch
from scipy.optimize import minimize
from .pid import PIDController

class PIDAutotuner:
    """
    Autotunes PID parameters (Kp, Ki, Kd) by running virtual simulations
    inside the learned JEPA World Model.
    Uses Nelder-Mead derivative-free optimization from scipy.
    """
    def __init__(self, world_model, temp_stats, window_length=10):
        self.world_model = world_model
        self.temp_mean = temp_stats['temp_mean']
        self.temp_std = temp_stats['temp_std']
        self.window_length = window_length

    def run_virtual_rollout(self, kp, ki, kd, target_temp, steps=120, initial_temp=25.0):
        """
        Simulates the closed-loop system entirely in the JEPA latent space.
        Returns:
            temps (list): predicted temperatures in deg C
            pwms (list): calculated PWM values
        """
        self.world_model.eval()
        pid = PIDController(kp=kp, ki=ki, kd=kd, target=target_temp)
        
        temps = [initial_temp]
        pwms = []
        
        # Initialize context window with initial_temp (normalized) and zero actions
        norm_init_temp = (initial_temp - self.temp_mean) / self.temp_std
        
        # Context window shape: (1, 2 * L) -> [T0, u0, T1, u1, ..., TL-1, uL-1]
        ctx = np.zeros(2 * self.window_length, dtype=np.float32)
        ctx[0::2] = norm_init_temp  # fill all T elements with norm_init_temp
        ctx[1::2] = 0.0             # fill all u elements with 0.0
        
        ctx_tensor = torch.tensor(ctx, dtype=torch.float32).unsqueeze(0) # (1, 2 * L)
        
        for step in range(steps):
            # 1. Read current temperature
            # For the first step, use initial_temp. For others, decode from the current state.
            if step == 0:
                current_temp = initial_temp
            else:
                with torch.no_grad():
                    state = self.world_model.encode_context(ctx_tensor)
                    norm_temp_pred = self.world_model.decode_temp(state).item()
                    current_temp = norm_temp_pred * self.temp_std + self.temp_mean
            
            # 2. Compute PID control action
            pwm = pid.compute(current_temp, dt=1.0)
            pwms.append(pwm)
            
            # 3. Predict next state in embedding space
            norm_pwm = pwm / 255.0
            action_tensor = torch.tensor([[norm_pwm]], dtype=torch.float32)
            
            with torch.no_grad():
                next_state_pred = self.world_model(ctx_tensor, action_tensor)
                norm_next_temp = self.world_model.decode_temp(next_state_pred).item()
            
            next_temp = norm_next_temp * self.temp_std + self.temp_mean
            temps.append(next_temp)
            
            # 4. Update the context window for the next iteration
            # Context window rolls forward: discard oldest (T, u), append (next_temp, norm_pwm)
            # Create a new context array
            new_ctx = np.zeros_like(ctx)
            # Shift the old elements
            new_ctx[0:-2] = ctx[2:]
            # Append new elements
            new_ctx[-2] = norm_next_temp
            new_ctx[-1] = norm_pwm
            
            ctx = new_ctx
            ctx_tensor = torch.tensor(ctx, dtype=torch.float32).unsqueeze(0)
            
        return temps[:-1], pwms

    def evaluate_cost(self, kp, ki, kd, target_temp, steps=120, initial_temp=25.0, 
                      w_tracking=2.0, w_overshoot=4.0, w_undershoot=10.0, w_actuator=0.005):
        """
        Calculates performance cost for a set of PID parameters.
        Penalizes:
          - tracking deviation from setpoint
          - overshoot beyond target
          - undershoot / droop below target
          - violent changes in PWM control signal
        """
        # Bounds constraints penalty
        if kp < 3.0 or kp > 35.0 or ki < 0.03 or ki > 1.5 or kd < 0.2 or kd > 6.0:
            return 1e6 # high penalty
            
        temps, pwms = self.run_virtual_rollout(kp, ki, kd, target_temp, steps, initial_temp)
        
        temps = np.array(temps)
        pwms = np.array(pwms)
        
        # 1. Tracking cost (sum of squared deviations after step 10 to ignore initial heat-up time)
        start_step = min(10, len(temps) // 2)
        tracking_err = temps[start_step:] - target_temp
        cost_tracking = np.mean(tracking_err ** 2)
        
        # 2. Overshoot cost
        max_temp = np.max(temps)
        overshoot = max(0.0, max_temp - target_temp)
        cost_overshoot = overshoot ** 2
        
        # 3. Undershoot / Thermal Droop cost (penalize staying below setpoint in settled window)
        final_window = temps[-min(40, len(temps)):]
        undershoot = np.maximum(0.0, target_temp - final_window)
        cost_undershoot = np.mean(undershoot ** 2)
        
        # 4. Actuator effort cost (penalize rapid changes in PWM)
        pwm_diffs = np.diff(pwms)
        cost_actuator = np.mean(pwm_diffs ** 2)
        
        total_cost = (
            w_tracking * cost_tracking + 
            w_overshoot * cost_overshoot + 
            w_undershoot * cost_undershoot + 
            w_actuator * cost_actuator
        )
        return total_cost

    def tune(self, target_temp, initial_temp=25.0, steps=120, x0=[15.0, 0.15, 2.5]):
        """
        Runs the Nelder-Mead optimizer to find Kp, Ki, Kd using a smooth sigmoid mapping.
        x0: Initial guess [Kp, Ki, Kd]
        """
        print(f"Starting autotuning search on JEPA World Model for target: {target_temp:.1f}C (Initial temp: {initial_temp:.1f}C)...")
        
        # Sigmoid mapping helper functions
        def sigmoid(v):
            return 1.0 / (1.0 + np.exp(-np.clip(v, -20.0, 20.0)))
            
        def logit(p):
            p = np.clip(p, 1e-5, 1.0 - 1e-5)
            return np.log(p / (1.0 - p))
            
        # Define realistic physical mapping bounds (prevents heater starvation & over-damping)
        min_kp, max_kp = 5.0, 35.0
        min_ki, max_ki = 0.05, 1.2
        min_kd, max_kd = 0.5, 6.0
        
        # Convert initial guess to logit space
        kp_init, ki_init, kd_init = x0
        kp_init = np.clip(kp_init, min_kp, max_kp)
        ki_init = np.clip(ki_init, min_ki, max_ki)
        kd_init = np.clip(kd_init, min_kd, max_kd)
        
        kp_p = (kp_init - min_kp) / (max_kp - min_kp)
        ki_p = (ki_init - min_ki) / (max_ki - min_ki)
        kd_p = (kd_init - min_kd) / (max_kd - min_kd)
        
        x0_logit = [logit(kp_p), logit(ki_p), logit(kd_p)]
        
        def loss_fn(x):
            # Map logit parameters back to linear space
            kp = min_kp + (max_kp - min_kp) * sigmoid(x[0])
            ki = min_ki + (max_ki - min_ki) * sigmoid(x[1])
            kd = min_kd + (max_kd - min_kd) * sigmoid(x[2])
            return self.evaluate_cost(kp, ki, kd, target_temp, steps, initial_temp)

        res = minimize(
            loss_fn, 
            x0=x0_logit, 
            method='Nelder-Mead', 
            options={'xatol': 1e-3, 'disp': True, 'maxiter': 200}
        )
        
        # Decode the optimal parameters
        best_kp = min_kp + (max_kp - min_kp) * sigmoid(res.x[0])
        best_ki = min_ki + (max_ki - min_ki) * sigmoid(res.x[1])
        best_kd = min_kd + (max_kd - min_kd) * sigmoid(res.x[2])
        
        print(f"Optimal parameters found: Kp={best_kp:.4f}, Ki={best_ki:.4f}, Kd={best_kd:.4f}")
        return best_kp, best_ki, best_kd
