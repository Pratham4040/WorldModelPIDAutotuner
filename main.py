import argparse
import os
import time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from world_model.simulator import ThermalSimulator
from world_model.client import ESP32Client
from world_model.pid import PIDController
from world_model.model import JEPAWorldModel
from world_model.dataset import prepare_data, TimeSeriesDataset
from world_model.trainer import JEPATrainer
from world_model.autotuner import PIDAutotuner

def ensure_dirs():
    """Ensure directory structure exists."""
    for d in ['data', 'models', 'reports']:
        os.makedirs(d, exist_ok=True)

def run_chamber_loop(hardware, pid, duration, target, excite_dynamics=False):
    """
    Runs the temperature control loop for a given duration with robust safety features.
    hardware: either ThermalSimulator or ESP32Client
    pid: PIDController instance
    duration: run time in seconds
    excite_dynamics: if True, toggles setpoint by +/- 1.5C periodically.
    """
    records = []
    
    # Initialize setpoint
    current_target = target
    pid.set_target(current_target)
    pid.reset()
    
    print(f"Starting loop for {duration} seconds with target {target}C...")
    
    # Safety tracking variables
    consecutive_read_errors = 0
    max_consecutive_errors = 3
    soft_temp_limit = 38.5  # Soft safety cutoff below hardware 39.0C
    
    # Thermal runaway tracking lists
    temp_history = []
    pwm_history = []
    runaway_window = 60
    
    for step in range(duration):
        loop_start = time.time()
        
        # 1. Read temperature
        temp, safety_active = hardware.read_temp()
        
        # Check for read failures
        if temp is None:
            print(f"[Warning] Connection lost or read failed. Safely forcing heater OFF (PWM=0)...")
            if isinstance(hardware, ESP32Client):
                hardware.set_pwm(0)
            
            # Reconnection loop
            reconnect_success = False
            max_reconnect_attempts = 30
            for attempt in range(1, max_reconnect_attempts + 1):
                print(f"  Attempting to reconnect and read sensor... ({attempt}/{max_reconnect_attempts})")
                time.sleep(2.0)
                temp, safety_active = hardware.read_temp()
                if temp is not None:
                    print(f"[Info] Reconnection successful! Resuming temperature control loop.")
                    reconnect_success = True
                    break
            
            if not reconnect_success:
                print(f"[CRITICAL SAFETY] Connection lost for over 60 seconds. Aborting loop!")
                break
            
            # Reset error tracking after successful reconnection inside the loop
            consecutive_read_errors = 0
            continue
        else:
            consecutive_read_errors = 0  # reset counter on success
            
        # Check soft safety temperature limit
        over_safety = (temp >= soft_temp_limit) or safety_active
        if over_safety:
            if step % 10 == 0:
                print(f"[SAFETY OVERRIDE] Temp ({temp:.2f}C >= {soft_temp_limit}C) or safety flag active! Forcing PWM=0 and continuing loop to monitor cooling...")
            pwm = 0.0
        else:
            # 2. Excite dynamics (toggle target by +/- 0.8C every 45 seconds to capture smooth heating/cooling phases)
            if excite_dynamics:
                # Shift setpoint every 45s across 3 phases: target, target + 0.8, target - 0.8
                phase = (step // 45) % 3
                if phase == 0:
                    current_target = target
                elif phase == 1:
                    current_target = target + 0.8
                else:
                    current_target = target - 0.8
                pid.set_target(current_target)
                
            # 3. Compute control action
            pwm = pid.compute(temp, dt=1.0)
            
            # Inject random dither during excitation to explore full state-action space
            if excite_dynamics:
                dither = np.random.uniform(-10.0, 10.0)
                pwm = max(0.0, min(255.0, pwm + dither))
        
        # 4. Write action to hardware
        if isinstance(hardware, ThermalSimulator):
            _, applied_pwm = hardware.step(pwm)
        else:
            success, applied_pwm = hardware.set_pwm(pwm)
            if not success:
                print(f"[Warning] Failed to write PWM to ESP32!")
                
        # 5. Record data
        records.append({
            'step': step,
            'temperature': temp,
            'pwm': applied_pwm,
            'target': current_target
        })
        
        # 6. Thermal runaway detection (only after we have enough history and step >= 30)
        temp_history.append(temp)
        pwm_history.append(applied_pwm)
        if len(temp_history) > runaway_window:
            temp_history.pop(0)
            pwm_history.pop(0)
            
        if step >= runaway_window:
            avg_pwm = np.mean(pwm_history)
            temp_change = temp_history[-1] - temp_history[0]
            # If we are heating at high power (avg PWM > 128), far from setpoint, but temperature isn't rising
            if avg_pwm > 180.0 and temp_change <= 0.1 and (current_target - temp) > 1.5:
                print(f"\n[CRITICAL SAFETY] THERMAL RUNAWAY DETECTED!")
                print(f"  Heater is running at high power (avg PWM: {avg_pwm:.1f}) but temperature rose only {temp_change:.2f}C in the last {runaway_window} seconds.")
                print(f"  Sensor may be detached or heater failed. Forcing heater OFF!")
                if isinstance(hardware, ESP32Client):
                    hardware.set_pwm(0)
                break
        
        # Log to terminal periodically
        if step % 10 == 0 or step == duration - 1:
            print(f"  Step {step:03d}/{duration:03d} | Temp: {temp:.2f}C | Setpoint: {current_target:.1f}C | PWM: {applied_pwm}")
            
        # 7. Sleep to maintain exactly 1Hz rate (1 second interval)
        if not isinstance(hardware, ThermalSimulator):
            elapsed = time.time() - loop_start
            sleep_time = max(0.0, 1.0 - elapsed)
            time.sleep(sleep_time)
        
    return pd.DataFrame(records)

def main():
    parser = argparse.ArgumentParser(description="End-to-End JEPA World Model PID Autotuning Pipeline")
    parser.add_argument('--mode', type=str, choices=['sim', 'real'], default='sim',
                        help="Run using simulator or physical hardware (default: sim)")
    parser.add_argument('--ip', type=str, default=None,
                        help="IP address of physical ESP32 (required if mode is real)")
    parser.add_argument('--target', type=float, default=35.0,
                        help="Target setpoint temperature to optimize for (default: 35.0)")
    parser.add_argument('--collect-time', type=int, default=180,
                        help="Duration in seconds for initial bootstrap data collection (default: 180)")
    parser.add_argument('--cycle-interval', type=int, default=300,
                        help="Duration in seconds per online refinement cycle (default: 300 = 5 min)")
    parser.add_argument('--conv-thresh', type=float, default=0.06,
                        help="Temperature StdDev threshold (deg C) to declare parameter convergence (default: 0.06)")
    parser.add_argument('--epochs', type=int, default=20,
                        help="Number of epochs for initial model training (default: 20)")
    parser.add_argument('--batch-size', type=int, default=32,
                        help="Batch size for model training (default: 32)")
    parser.add_argument('--lr', type=float, default=1e-3,
                        help="Learning rate for model training (default: 1e-3)")
    parser.add_argument('--window', type=int, default=15,
                        help="History window size L in seconds (default: 15)")
    parser.add_argument('--kp-init', type=float, default=6.0,
                        help="Initial PID Kp value for data collection (default: 6.0)")
    parser.add_argument('--ki-init', type=float, default=0.05,
                        help="Initial PID Ki value for data collection (default: 0.05)")
    parser.add_argument('--kd-init', type=float, default=2.0,
                        help="Initial PID Kd value for data collection (default: 2.0)")
    
    args = parser.parse_args()
    ensure_dirs()
    
    if args.mode == 'real' and args.ip is None:
        parser.error("--ip is required when running in physical 'real' mode!")
        
    print("=====================================================================")
    print("      JEPA WORLD MODEL & PID AUTOTUNER AUTOMATED PIPELINE STARTED     ")
    print("=====================================================================")
    print(f"Mode: {args.mode.upper()}")
    if args.mode == 'real':
        print(f"ESP32 IP: {args.ip}")
    print(f"Target Temperature: {args.target}C")
    print(f"Initial PID parameters: Kp={args.kp_init}, Ki={args.ki_init}, Kd={args.kd_init}")
    print("=====================================================================\n")
    
    # Initialize hardware interface
    if args.mode == 'sim':
        print("Initializing thermal simulator...")
        hardware = ThermalSimulator(ambient_temp=24.0, sensor_noise_std=0.02)
    else:
        print(f"Initializing connection to ESP32 at {args.ip}...")
        hardware = ESP32Client(args.ip)
        
    # Target safety limit check
    if args.target > 38.0:
        print(f"[CRITICAL SAFETY] Target temperature cannot be set above 38.0C for hardware protection! Setpoint requested: {args.target}C. Exiting.")
        return
        
    # Test connection/read ambient
    t_init, safety = hardware.read_temp()
    if t_init is None or safety:
        print("[CRITICAL] Could not read temperature from hardware/simulator. Exiting.")
        return
        
    # Initial sensor sanity check
    if t_init < 10.0 or t_init > 40.0:
        print(f"[CRITICAL SAFETY] Initial temperature reading {t_init:.2f}C is outside plausible bounds (10C - 40C). Sensor may be malfunctioning or disconnected. Exiting.")
        return
        
    print(f"Initial temperature reading: {t_init:.2f}C (Safety flag: {safety})\n")
    
    data_path = "data/collected_data.csv"
    model_path = "models/jepa_model.pt"
    
    try:
        # =====================================================================
        # PRE-HEAT PHASE & THERMAL SETTLING
        # =====================================================================
        preheat_threshold = args.target - 2.5
        if t_init < preheat_threshold:
            print("---------------------------------------------------------------------")
            print(f" PRE-HEAT PHASE: Warming chamber to {preheat_threshold:.1f}C before tuning...")
            print("---------------------------------------------------------------------")
            
            preheat_duration = 300
            for step in range(preheat_duration):
                temp, safety_active = hardware.read_temp()
                if temp is None or safety_active or temp >= preheat_threshold:
                    if temp is not None and temp >= preheat_threshold:
                        print(f"Pre-heat target reached! Current temperature: {temp:.2f}C.")
                    break
                
                pwm = 255.0  # Apply maximum heat to reach preheat threshold rapidly (Bang-Bang Control)
                if isinstance(hardware, ThermalSimulator):
                    hardware.step(pwm)
                else:
                    hardware.set_pwm(pwm)
                    
                if step % 10 == 0:
                    print(f"  Preheating | Temp: {temp:.2f}C | Threshold: {preheat_threshold:.1f}C | PWM: {int(pwm)}")
                
                if not isinstance(hardware, ThermalSimulator):
                    time.sleep(1.0)
            
            # Settling phase: turn off heater and wait for thermal inertia overshoot to peak and cool down
            print("\n---------------------------------------------------------------------")
            print(f" PRE-HEAT SETTLING: Waiting up to 90s for thermal inertia to settle below {args.target - 0.5:.1f}C...")
            print("---------------------------------------------------------------------")
            if isinstance(hardware, ESP32Client):
                hardware.set_pwm(0)
                
            settle_target = args.target - 0.5
            max_settle_steps = 90
            for step in range(max_settle_steps):
                temp, safety_active = hardware.read_temp()
                if temp is None or safety_active:
                    print("[Warning] Sensor error or safety active during settling!")
                    break
                
                # Check if temperature has peaked and cooled down below settle_target
                if step >= 15 and temp <= settle_target:
                    print(f"Chamber settled cleanly at {temp:.2f}C! Ready for Data Collection.\n")
                    break
                    
                if step % 10 == 0:
                    print(f"  Settling | PWM: 0 | Current Temp: {temp:.2f}C | Target: < {settle_target:.1f}C")
                    
                if not isinstance(hardware, ThermalSimulator):
                    time.sleep(1.0)

        # =====================================================================
        # PHASE 1: Data Collection
        # =====================================================================
        print("---------------------------------------------------------------------")
        print(" PHASE 1: Running Data Collection (PC-in-the-Loop)...")
        print("---------------------------------------------------------------------")
        # Initialize PID with default gains
        init_pid = PIDController(kp=args.kp_init, ki=args.ki_init, kd=args.kd_init, target=args.target)
        
        # Collect data while exciting the thermal dynamics (toggling setpoint)
        df_train = run_chamber_loop(hardware, init_pid, args.collect_time, args.target, excite_dynamics=True)
        
        # Turn off heater cleanly
        if args.mode == 'sim':
            hardware.reset(df_train['temperature'].iloc[-1])
        else:
            hardware.set_pwm(0)
            
        min_required_records = int(args.collect_time * 0.75)
        if len(df_train) < min_required_records:
            print(f"\n[CRITICAL ERROR] Phase 1 Data Collection was interrupted early by safety trip or connection error!")
            print(f"Collected only {len(df_train)}/{args.collect_time} records (minimum required: {min_required_records}).")
            print("Pipeline aborted to prevent training on incomplete/corrupted data.\n")
            if args.mode == 'real':
                hardware.set_pwm(0)
            return
            
        df_train.to_csv(data_path, index=False)
        print(f"Dataset successfully saved to: {data_path} ({len(df_train)} records)\n")
        
        # Allow system to cool down slightly/pause between phases
        print("Waiting 3 seconds for system transition...")
        time.sleep(3.0)
        
        # =====================================================================
        # PHASE 2: JEPA World Model Training
        # =====================================================================
        print("---------------------------------------------------------------------")
        print(" PHASE 2: Training JEPA World Model on Collected Data...")
        print("---------------------------------------------------------------------")
        
        # Prepare dataset windows
        contexts, actions, targets, target_temps, stats = prepare_data(data_path, window_length=args.window)
        
        # Split into Train/Val sets (80/20)
        num_samples = len(contexts)
        split_idx = int(num_samples * 0.8)
        
        # Shuffle indices
        indices = np.random.permutation(num_samples)
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        train_dataset = TimeSeriesDataset(
            contexts[train_indices], actions[train_indices], 
            targets[train_indices], target_temps[train_indices]
        )
        val_dataset = TimeSeriesDataset(
            contexts[val_indices], actions[val_indices], 
            targets[val_indices], target_temps[val_indices]
        )
        
        # Instantiate JEPA Model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = JEPAWorldModel(window_length=args.window, latent_dim=8).to(device)
        model.temp_mean = stats['temp_mean']
        model.temp_std = stats['temp_std']
        
        # Train
        trainer = JEPATrainer(model, lr=args.lr, weight_decay=1e-4, ema_decay=0.99, lambda_dec=1.0)
        history = trainer.train(train_dataset, val_dataset, epochs=args.epochs, batch_size=args.batch_size, device=device)
        
        # Save weights and normalization stats
        torch.save({
            'state_dict': model.state_dict(),
            'stats': stats,
            'window_length': args.window
        }, model_path)
        print(f"JEPA World Model weights saved to: {model_path}")
        
        # Save training loss plots
        plt.figure(figsize=(10, 5))
        plt.plot(history['train_total'], label='Train Total')
        plt.plot(history['val_total'], label='Val Total')
        plt.plot(history['train_jepa'], '--', label='Train JEPA (Latent)')
        plt.plot(history['val_jepa'], '--', label='Val JEPA (Latent)')
        plt.plot(history['train_dec'], ':', label='Train Decoder (Grounding)')
        plt.plot(history['val_dec'], ':', label='Val Decoder (Grounding)')
        plt.title('JEPA World Model Training Losses')
        plt.xlabel('Epochs')
        plt.ylabel('Loss (MSE)')
        plt.grid(True)
        plt.legend()
        plt.savefig('reports/training_loss.png')
        plt.close()
        print("Training loss curve saved to: reports/training_loss.png\n")
        
        # =====================================================================
        # PHASE 3: PID Autotuning on World Model
        # =====================================================================
        print("---------------------------------------------------------------------")
        print(" PHASE 3: Autotuning PID Parameters inside Virtual World Model...")
        print("---------------------------------------------------------------------")
        autotuner = PIDAutotuner(model, stats, window_length=args.window)
        
        # Run optimization
        best_kp, best_ki, best_kd = autotuner.tune(
            target_temp=args.target, 
            initial_temp=args.target - 2.0, 
            steps=300, 
            x0=[args.kp_init, args.ki_init, args.kd_init]
        )
        
        # Evaluate virtual rollouts
        v_temps_init, v_pwms_init = autotuner.run_virtual_rollout(
            args.kp_init, args.ki_init, args.kd_init, args.target, steps=300, initial_temp=t_init
        )
        v_temps_opt, v_pwms_opt = autotuner.run_virtual_rollout(
            best_kp, best_ki, best_kd, args.target, steps=300, initial_temp=t_init
        )
        
        # Save rollout plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        ax1.plot(v_temps_init, label=f'Initial PID (Kp={args.kp_init:.1f}, Ki={args.ki_init:.2f})', color='red', alpha=0.7)
        ax1.plot(v_temps_opt, label=f'Optimized PID (Kp={best_kp:.2f}, Ki={best_ki:.3f}, Kd={best_kd:.2f})', color='green')
        ax1.axhline(args.target, color='blue', linestyle='--', label='Setpoint')
        ax1.set_ylabel('Temperature (C)')
        ax1.set_title('Virtual Chamber Rollout inside JEPA World Model')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(v_pwms_init, label='Initial PID PWM', color='red', linestyle='--', alpha=0.7)
        ax2.plot(v_pwms_opt, label='Optimized PID PWM', color='green')
        ax2.set_xlabel('Steps (Seconds)')
        ax2.set_ylabel('PWM Duty Cycle (0-255)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('reports/autotune_rollout.png')
        plt.close()
        print("Virtual simulation rollout comparison saved to: reports/autotune_rollout.png\n")
        
        # =====================================================================
        # PHASE 4: Online Continuous Adaptive Autotuning Loop
        # =====================================================================
        print("---------------------------------------------------------------------")
        print(" PHASE 4: Launching Online Continuous Adaptive Autotuning Loop...")
        print(" System will continuously regulate temperature while fine-tuning the")
        print(f" JEPA World Model on live data and updating PID gains every {args.cycle_interval // 60} min.")
        print(" Press Ctrl+C at any time to stop and save the final parameters.")
        print("---------------------------------------------------------------------\n")
        
        cycle = 1
        low_var_consecutive = 0
        
        while True:
            print("=====================================================================")
            print(f" ONLINE AUTOTUNING CYCLE #{cycle}")
            print(f" Active Gains: Kp={best_kp:.4f}, Ki={best_ki:.4f}, Kd={best_kd:.4f}")
            print(f" Interval Duration: {args.cycle_interval}s ({args.cycle_interval // 60} minutes)")
            print("=====================================================================")
            
            # 1. Instantiate PID with current optimal gains
            active_pid = PIDController(kp=best_kp, ki=best_ki, kd=best_kd, target=args.target)
            
            # 2. Run active regulation loop for cycle_interval seconds
            df_cycle = run_chamber_loop(hardware, active_pid, args.cycle_interval, args.target, excite_dynamics=False)
            
            if df_cycle.empty or len(df_cycle) < 10:
                print("[Warning] Cycle data collection interrupted or empty. Retrying cycle...")
                continue
                
            # Append new cycle data to global accumulated dataset
            df_train = pd.concat([df_train, df_cycle], ignore_index=True)
            df_train.to_csv(data_path, index=False)
            
            # 3. Calculate cycle performance metrics over settled step window (last 100 steps)
            recent_temps = df_cycle['temperature'].iloc[-min(100, len(df_cycle)):]
            recent_err = recent_temps - args.target
            cycle_rmse = np.sqrt(np.mean(recent_err**2))
            cycle_std = np.std(recent_temps)
            
            print(f"\n[Cycle #{cycle} Performance Summary]")
            print(f"  Settled Fluctuation (StdDev): {cycle_std:.4f} C")
            print(f"  RMSE to target ({args.target}C):      {cycle_rmse:.4f} C")
            print(f"  Active Gains Used:           Kp={best_kp:.4f}, Ki={best_ki:.4f}, Kd={best_kd:.4f}")
            
            # 4. Check for Parameter Convergence
            if cycle_std <= args.conv_thresh and cycle_rmse <= args.conv_thresh * 3.0:
                low_var_consecutive += 1
                print(f"  [CONVERGENCE CHECK] Low temperature fluctuation detected ({low_var_consecutive}/2 consecutive cycles)!")
            else:
                low_var_consecutive = 0
                
            if low_var_consecutive >= 2:
                print("\n*********************************************************************")
                print(f" 🎉 CONVERGENCE ACHIEVED AT CYCLE #{cycle}!")
                print(f" Temperature fluctuation is under target threshold ({cycle_std:.4f} C <= {args.conv_thresh:.2f} C).")
                print(f" FINAL OPTIMAL PID GAINS FOUND:")
                print(f"    Kp = {best_kp:.4f}")
                print(f"    Ki = {best_ki:.4f}")
                print(f"    Kd = {best_kd:.4f}")
                print(" Copy these parameters into your Arduino sketch for standalone mode!")
                print("*********************************************************************\n")
                
                # Keep running standalone regulation with final parameters
                print("Entering continuous standalone regulation mode with final optimal parameters...")
                final_pid = PIDController(kp=best_kp, ki=best_ki, kd=best_kd, target=args.target)
                run_chamber_loop(hardware, final_pid, 99999999, args.target, excite_dynamics=False)
                break
                
            # 5. Online Fine-Tuning of JEPA World Model on accumulated dataset
            print(f"\n---------------------------------------------------------------------")
            print(f" [ONLINE TRAINING] Fine-tuning JEPA World Model on {len(df_train)} total samples...")
            print("---------------------------------------------------------------------")
            
            contexts, actions, targets, target_temps, stats = prepare_data(data_path, window_length=args.window)
            num_samples = len(contexts)
            split_idx = int(num_samples * 0.8)
            indices = np.random.permutation(num_samples)
            
            train_dataset = TimeSeriesDataset(
                contexts[indices[:split_idx]], actions[indices[:split_idx]], 
                targets[indices[:split_idx]], target_temps[indices[:split_idx]]
            )
            val_dataset = TimeSeriesDataset(
                contexts[indices[split_idx:]], actions[indices[split_idx:]], 
                targets[indices[split_idx:]], target_temps[indices[split_idx:]]
            )
            
            model.temp_mean = stats['temp_mean']
            model.temp_std = stats['temp_std']
            history = trainer.train(train_dataset, val_dataset, epochs=10, batch_size=args.batch_size, device=device)
            
            # Save updated weights
            torch.save({
                'state_dict': model.state_dict(),
                'stats': stats,
                'window_length': args.window
            }, model_path)
            
            # 6. Online Autotuning on updated World Model
            print(f"---------------------------------------------------------------------")
            print(f" [ONLINE AUTOTUNING] Searching for refined PID parameters...")
            print("---------------------------------------------------------------------")
            autotuner = PIDAutotuner(model, stats, window_length=args.window)
            new_kp, new_ki, new_kd = autotuner.tune(
                target_temp=args.target, 
                initial_temp=args.target - 2.0, 
                steps=300, 
                x0=[best_kp, best_ki, best_kd]
            )
            
            print(f"\n[Gains Updated for Cycle #{cycle+1}] Kp: {best_kp:.4f} -> {new_kp:.4f} | Ki: {best_ki:.4f} -> {new_ki:.4f} | Kd: {best_kd:.4f} -> {new_kd:.4f}\n")
            best_kp, best_ki, best_kd = new_kp, new_ki, new_kd
            cycle += 1
        
    except KeyboardInterrupt:
        print("\n[CRITICAL] Execution interrupted by user! Safety override: forcing heater OFF.")
        if args.mode == 'real':
            # Send safety command
            try:
                hardware.set_pwm(0)
                print("Safety command sent successfully: PWM=0")
            except Exception as e:
                print(f"Failed to send safety command: {e}")
        else:
            hardware.reset()
            
if __name__ == '__main__':
    main()
