import numpy as np
import h5py
from scipy.signal import hilbert
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import r2_score
import seaborn as sns
import os
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
import pandas as pd


class WAISPulser:
    """Information about WAIS pulsing settings during ANITA-4 flight"""

    def __init__(self):
        # Constants from WAIS documentation
        self.VPOL_PULSER_VOLTAGE = 5000  # 5 kV for VPol (FID-FPG61-PNK)
        self.HPOL_PULSER_VOLTAGE = 4800  # 4.8 kV for HPol (FID-FPM10-1-PNP)
        self.CABLE_LENGTH = 15.24  # 50 ft in meters
        self.CABLE_VELOCITY_FACTOR = 0.87  # for LMR-600
        self.SYSTEM_IMPEDANCE = 50.0  # ohms
        self.GPS_TIMING_JITTER = 250e-9  # 250 ns

        # WAIS pulser location
        self.WAIS_LAT = -79.46811  # -79° 28' 5.2190"
        self.WAIS_LON = -112.05926  # -112° 03' 33.329"
        self.WAIS_ALT = 1779.80  # meters

        # Create timing and attenuation lookup tables from pulser notes
        self._init_attenuation_data()

    def _init_attenuation_data(self):
        """Initialize attenuation and timing data from WAIS pulser notes"""
        # Format: [approx_datetime, v_attenuation, h_attenuation, alignment]
        # All datetimes in NZDT as per the pulser notes
        # Timestamp format: 'YYYY-MM-DD HH:MM'
        self.attenuation_data = [
            # Day 1 (Dec 9)
            {'time': '2016-12-09 11:32', 'v_atten': 20, 'h_atten': 20, 'alignment': 'V/H', 'az': 78.4, 'el': 0.7},
            {'time': '2016-12-09 16:08', 'v_atten': 30, 'h_atten': 20, 'alignment': 'V/H', 'az': 79.7, 'el': 1.7},
            {'time': '2016-12-09 19:34', 'v_atten': 30, 'h_atten': 30, 'alignment': 'V/H', 'az': 86.8, 'el': 3.8},
            {'time': '2016-12-09 23:30', 'v_atten': 40, 'h_atten': 40, 'alignment': '45°', 'az': 84.0, 'el': 5.0},
            # Day 2 (Dec 10)
            {'time': '2016-12-10 01:50', 'v_atten': 30, 'h_atten': 30, 'alignment': '45°', 'az': 84.0, 'el': 5.0},
            {'time': '2016-12-10 08:00', 'v_atten': 40, 'h_atten': 40, 'alignment': 'V/H', 'az': 50.0, 'el': 26.0},
            {'time': '2016-12-10 15:22', 'v_atten': 30, 'h_atten': 30, 'alignment': 'V/H', 'az': 246.2, 'el': 8.8},
            {'time': '2016-12-10 16:55', 'v_atten': 40, 'h_atten': 40, 'alignment': 'V/H', 'az': 235.0, 'el': 8.5},
            {'time': '2016-12-10 18:56', 'v_atten': 30, 'h_atten': 30, 'alignment': 'V/H', 'az': 224.0, 'el': 8.3},
            # Day 3 (Dec 11)
            {'time': '2016-12-11 12:11', 'v_atten': 20, 'h_atten': 20, 'alignment': 'V/H', 'az': 246.0, 'el': 2.5},
            {'time': '2016-12-11 13:13', 'v_atten': 0, 'h_atten': 0, 'alignment': 'V/H', 'az': 239.6, 'el': 1.2}
        ]

        # Convert to datetime objects for easier lookup
        for entry in self.attenuation_data:
            entry['datetime'] = pd.to_datetime(entry['time'])

        # Sort by time
        self.attenuation_data = sorted(self.attenuation_data, key=lambda x: x['datetime'])

    def get_attenuation(self, event_time):
        """Get the attenuation values for a specific event time

        Args:
            event_time: Unix timestamp or datetime object

        Returns:
            dict: Attenuation settings and orientation information
        """
        # Convert timestamp to datetime if needed
        if isinstance(event_time, (int, float)):
            dt = pd.to_datetime(event_time, unit='s')
        else:
            dt = pd.to_datetime(event_time)

        # Find the correct attenuation entry (the last one before this event)
        attenuation_entry = None
        for entry in self.attenuation_data:
            if entry['datetime'] <= dt:
                attenuation_entry = entry
            else:
                break

        # If no entry found (event is before first attenuation change), use the first entry
        if attenuation_entry is None and len(self.attenuation_data) > 0:
            attenuation_entry = self.attenuation_data[0]

        return attenuation_entry if attenuation_entry else {
            'v_atten': 20, 'h_atten': 20, 'alignment': 'V/H',
            'az': 0, 'el': 0, 'datetime': None
        }

    def calculate_real_energy(self, voltage, pol='V', attenuation=0):
        """Calculate real energy from pulser voltage, considering attenuation

        Args:
            voltage: Measured voltage
            pol: Polarization ('V' or 'H')
            attenuation: Attenuation in dB

        Returns:
            float: Energy in Joules
        """
        # Constants for real energy calculation
        PULSE_WIDTH = 1e-9  # 1 ns pulse width

        # Get reference voltage based on polarization
        reference_voltage = self.VPOL_PULSER_VOLTAGE if pol == 'V' else self.HPOL_PULSER_VOLTAGE

        # Calculate peak power (P = V²/R) with attenuation adjustment
        attenuation_factor = 10**(attenuation/10)  # Convert dB to linear factor
        peak_power_watts = (reference_voltage ** 2) / (self.SYSTEM_IMPEDANCE * attenuation_factor)

        # Calculate energy (E = P * t)
        energy_joules = peak_power_watts * PULSE_WIDTH

        return energy_joules

    def calculate_fspl(self, distance, frequency):
        """Calculate free space path loss"""
        c = 3e8  # speed of light
        return (4 * np.pi * distance * frequency / c) ** 2

    def calculate_timing_offset(self, is_hpol):
        """Calculate timing offset based on pulser configuration"""
        return -10e-6 if is_hpol else 0  # -10 μs for HPol, 0 for VPol

    def calculate_cable_delay(self):
        """Calculate cable delay for 50ft LMR-600"""
        return self.CABLE_LENGTH / (self.CABLE_VELOCITY_FACTOR * 3e8)

    def calculate_distance(self, anita_lat, anita_lon, anita_alt):
        """Calculate distance between ANITA and WAIS pulser"""
        R = 6371000  # Earth radius in meters

        # Convert to radians
        lat1, lon1 = np.radians(self.WAIS_LAT), np.radians(self.WAIS_LON)
        lat2, lon2 = np.radians(anita_lat), np.radians(anita_lon)

        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        ground_distance = R * c

        # Include altitude difference
        height_diff = anita_alt - self.WAIS_ALT
        return np.sqrt(ground_distance**2 + height_diff**2)


class WAISEnergyProcessor:
    """Processes WAIS signals to predict source energy"""

    def __init__(self):
        # Initialize WAIS pulser info
        self.pulser = WAISPulser()
    
    def process_waveform(self, voltages, distance, event_time, channel_pol='V', frequency=300e6):
        """Extract features from voltage waveform with proper physics considerations
    
        Args:
            voltages: Array of voltage measurements
            distance: Distance from ANITA to WAIS in meters
            event_time: Timestamp for attenuation lookup
            channel_pol: Channel polarization ('V' or 'H')
            frequency: Signal frequency in Hz
    
        Returns:
            numpy.ndarray: Feature vector
        """
        if len(voltages) == 0:
            return np.zeros(6)  # Return empty feature vector
    
        # Get attenuation settings for this event time
        atten_info = self.pulser.get_attenuation(event_time)
        attenuation = atten_info['v_atten'] if channel_pol == 'V' else atten_info['h_atten']
        alignment = atten_info['alignment']
        azimuth = atten_info['az']
        elevation = atten_info['el']
    
        # Calculate signal envelope
        analytic_signal = hilbert(voltages)
        envelope = np.abs(analytic_signal)
    
        # Convert voltage to power
        instantaneous_power = envelope ** 2 / (2 * self.pulser.SYSTEM_IMPEDANCE)
    
        # Account for free space path loss
        fspl = self.pulser.calculate_fspl(distance, frequency)
        source_power = instantaneous_power * fspl
    
        # Calculate features
        peak_power = np.max(source_power)
        mean_power = np.mean(source_power)
        total_energy = np.sum(source_power) * (1 / frequency)
        waveform_std = np.std(source_power)
    
        # Calculate real energy
        real_energy = self.pulser.calculate_real_energy(
            np.max(envelope),
            pol=channel_pol,
            attenuation=attenuation
        )
    
        # Calculate theoretical maximum energy based on the reference voltage and attenuation
        reference_voltage = (self.pulser.VPOL_PULSER_VOLTAGE if channel_pol == 'V'
                           else self.pulser.HPOL_PULSER_VOLTAGE)
        
        # Calculate the pulse duration (assuming 1ns for WAIS)
        pulse_duration = 1e-9  
        
        # Convert attenuation from dB to linear scale
        attenuation_factor = 10**(attenuation/10)
        
        # Calculate max theoretical power at the source (after attenuation)
        max_theoretical_power = reference_voltage**2 / (self.pulser.SYSTEM_IMPEDANCE * attenuation_factor)
        
        # Calculate max theoretical energy (Power * Time)
        max_theoretical_energy = max_theoretical_power * pulse_duration
        
        # Normalize energy using theoretical maximum (guarantees values between 0 and ~1)
        normalized_energy = total_energy / max_theoretical_energy
        
        # Return feature vector
        return np.array([
            peak_power,
            mean_power,
            normalized_energy,
            waveform_std,
            real_energy,
            attenuation  # Add attenuation as a feature
        ])


def get_wais_data(event_files, header_files):
    """Load WAIS calibration events with enhanced physics features

    Args:
        event_files: List of event HDF5 files
        header_files: List of corresponding header HDF5 files

    Returns:
        tuple: (features, source_energies, distances, event_times)
    """
    processor = WAISEnergyProcessor()
    features = []
    source_energies = []
    distances = []
    event_times = []
    attenuation_values = {'V': [], 'H': []}

    print("\nProcessing WAIS calibration data...")
    for event_file, header_file in zip(event_files, header_files):
        try:
            with h5py.File(event_file, 'r') as ef, h5py.File(header_file, 'r') as hf:
                run_number = int(event_file.split("event")[-1].split(".")[0])
                run_key = f'run{run_number}'

                if run_key not in ef or run_key not in hf:
                    print(f"Run {run_key} not found in files")
                    continue

                run_events = ef[run_key]
                run_header = hf[run_key]

                # Get ANITA position and time
                anita_lat = run_header['latitude'][:]
                anita_lon = run_header['longitude'][:]
                anita_alt = run_header['altitude'][:]

                # Get event timestamps for attenuation lookup
                if 'triggerTime' in run_header:
                    event_timestamps = run_header['triggerTime'][:]
                else:
                    # If no trigger time, use a placeholder (will use default attenuation)
                    event_timestamps = np.zeros_like(anita_lat)

                # Get WAIS calibration events
                is_wais = run_header['isWAIS'][:]
                wais_indices = np.where(is_wais)[0]

                print(f"\nProcessing {len(wais_indices)} WAIS events from {run_key}")

                for idx in tqdm(wais_indices):
                    try:
                        event_num = run_header['eventNumber'][idx]
                        ev_name = f'ev_{event_num}'

                        if ev_name in run_events:
                            # Get event timestamp
                            event_time = event_timestamps[idx]

                            # Calculate distance to ANITA
                            distance = processor.pulser.calculate_distance(
                                anita_lat[idx],
                                anita_lon[idx],
                                anita_alt[idx]
                            )

                            # Get attenuation settings for this event
                            atten_info = processor.pulser.get_attenuation(event_time)

                            # Process voltage data
                            event = run_events[ev_name]
                            voltages = np.array(event['data'])
                            event_features = []

                            # Calculate energies
                            vpol_energy = 0
                            hpol_energy = 0
                            vpol_real_energy = 0
                            hpol_real_energy = 0
                            n_vpol = 0
                            n_hpol = 0

                            for ant in range(48):
                                surf = ant // 4
                                chan = ant % 4
                                if surf < voltages.shape[0] and chan < voltages.shape[1]:
                                    pol = 'V' if chan in [0, 2] else 'H'

                                    # Calculate features including energy with attenuation
                                    ant_features = processor.process_waveform(
                                        voltages[surf, chan],
                                        distance=distance,
                                        event_time=event_time,
                                        channel_pol=pol
                                    )
                                    event_features.append(ant_features)

                                    # Accumulate energy by polarization
                                    if pol == 'V':
                                        vpol_energy += ant_features[2]  # normalized energy
                                        vpol_real_energy += ant_features[4]  # real energy
                                        n_vpol += 1
                                    else:
                                        hpol_energy += ant_features[2]
                                        hpol_real_energy += ant_features[4]
                                        n_hpol += 1

                            if len(event_features) == 48:
                                features.append(event_features)
                                distances.append(distance)
                                event_times.append(event_time)

                                # Calculate average energies
                                avg_vpol = vpol_energy / n_vpol if n_vpol > 0 else 0
                                avg_hpol = hpol_energy / n_hpol if n_hpol > 0 else 0
                                total_energy = (avg_vpol + avg_hpol) / 2

                                source_energies.append(total_energy)

                                # Store attenuation values for statistics
                                attenuation_values['V'].append(atten_info['v_atten'])
                                attenuation_values['H'].append(atten_info['h_atten'])

                    except Exception as e:
                        print(f"Error processing event {event_num}: {str(e)}")
                        continue

        except Exception as e:
            print(f"Error processing file {event_file}: {str(e)}")
            continue

    # Convert to arrays
    features = np.array(features)
    source_energies = np.array(source_energies)
    distances = np.array(distances)
    event_times = np.array(event_times)

    # Print statistics
    print("\nProcessed Data Statistics:")
    print("-" * 50)
    print(f"Total events processed: {len(source_energies)}")
    print(f"\nNormalized Energy statistics:")
    print(f"  Mean: {np.mean(source_energies):.6f}")
    print(f"  Std Dev: {np.std(source_energies):.6f}")
    print(f"  Min: {np.min(source_energies):.6f}")
    print(f"  Max: {np.max(source_energies):.6f}")

    print(f"\nReal Energy statistics (Joules):")
    print(f"  Mean: {np.mean(features[:,:,4]):.6e}")
    print(f"  Std Dev: {np.std(features[:,:,4]):.6e}")
    print(f"  Min: {np.min(features[:,:,4]):.6e}")
    print(f"  Max: {np.max(features[:,:,4]):.6e}")

    print(f"\nAttenuation statistics:")
    print(f"  V-pol: {np.unique(attenuation_values['V'])}")
    print(f"  H-pol: {np.unique(attenuation_values['H'])}")

    print(f"\nDistance range: {np.min(distances)/1000:.1f} km to {np.max(distances)/1000:.1f} km")

    return features, source_energies, distances, event_times


class WAISDataset(Dataset):
    """PyTorch dataset for WAIS source energy prediction with max normalization"""
    
    def __init__(self, features, source_energies, distances=None, event_times=None):
        # Convert to tensors
        features_tensor = torch.FloatTensor(features)
        
        # Normalize features
        self.feature_mean = torch.mean(features_tensor, dim=0, keepdim=True)
        self.feature_std = torch.std(features_tensor, dim=0, keepdim=True) + 1e-8
        self.features = (features_tensor - self.feature_mean) / self.feature_std
        
        # Store raw energy values
        self.source_energies_raw = torch.FloatTensor(source_energies)
        
        # Apply max normalization to source energies
        self.energy_min = self.source_energies_raw.min()
        self.energy_max = self.source_energies_raw.max()
        
        # Normalize to [0,1] range
        self.source_energies = (self.source_energies_raw - self.energy_min) / (self.energy_max - self.energy_min)
        
        # Store distances and event times if provided
        self.distances = torch.FloatTensor(distances) if distances is not None else None
        self.event_times = torch.FloatTensor(event_times) if event_times is not None else None
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        sample = {
            'features': self.features[idx],
            'energy': self.source_energies[idx]
        }
        
        if self.distances is not None:
            sample['distance'] = self.distances[idx]
            
        if self.event_times is not None:
            sample['time'] = self.event_times[idx]
            
        return sample
    
    def denormalize_energy(self, normalized_energy):
        """Convert normalized energy back to original scale"""
        return normalized_energy * (self.energy_max - self.energy_min) + self.energy_min
    
    def denormalize_features(self, normalized_features):
        """Convert normalized features back to original scale"""
        return normalized_features * self.feature_std + self.feature_mean
        

class EnhancedEnergyPredictor(nn.Module):
    """Enhanced neural network for WAIS pulser source energy prediction
    with attenuation awareness"""

    def __init__(self, input_features=6, hidden_size=64):
        super().__init__()

        # Process each antenna's signals
        self.antenna_processor = nn.Sequential(
            nn.Linear(input_features, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size//2)
        )

        # Attention mechanism for antenna importance
        self.attention = nn.Sequential(
            nn.Linear(hidden_size//2, hidden_size//4),
            nn.Tanh(),
            nn.Linear(hidden_size//4, 1)
        )

        # Final energy prediction layers
        self.energy_predictor = nn.Sequential(
            nn.Linear(hidden_size//2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size//2, 1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize network weights using Xavier initialization"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        batch_size = x.size(0)

        # Process all antennas
        x = x.view(-1, x.size(-1))  # (batch*48, features)
        x = self.antenna_processor(x)  # (batch*48, hidden_size//2)

        # Reshape for attention
        x = x.view(batch_size, 48, -1)  # (batch, 48, hidden_size//2)

        # Calculate attention weights
        attention_weights = self.attention(x)  # (batch, 48, 1)
        attention_weights = F.softmax(attention_weights, dim=1)

        # Apply attention
        weighted_features = (x * attention_weights).sum(dim=1)  # (batch, hidden_size//2)

        # Predict energy
        source_energy = self.energy_predictor(weighted_features)

        return source_energy.squeeze(), attention_weights.squeeze()

    def get_attention_map(self, x):
        """Get attention weights for visualization"""
        with torch.no_grad():
            _, attention = self.forward(x)
            return attention.cpu().numpy()


def create_improved_splits(dataset, train_size=0.8, val_size=0.1, seed=42):
    """Create data splits while maintaining energy distribution"""
    total_size = len(dataset)
    train_len = int(train_size * total_size)
    val_len = int(val_size * total_size)

    # Create splits using random indices
    indices = torch.randperm(total_size, generator=torch.Generator().manual_seed(seed))

    train_indices = indices[:train_len]
    val_indices = indices[train_len:train_len + val_len]
    test_indices = indices[train_len + val_len:]

    # Create subsets
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    return train_dataset, val_dataset, test_dataset


def train_model(model, train_loader, val_loader, epochs=50, device='cpu'):
    """Train the source energy prediction model"""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )

    # Initialize tracking
    best_val_loss = float('inf')
    best_model = None
    patience = 10
    patience_counter = 0
    train_metrics = {'mse': [], 'mae': [], 'r2': []}
    val_metrics = {'mse': [], 'mae': [], 'r2': []}

    print("\nTraining Enhanced Source Energy Predictor:")
    print(f"{'Epoch':>5} {'Train Loss':>12} {'Val Loss':>12} {'R²':>8} {'LR':>10}")
    print("-" * 50)

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_predictions = []
        train_targets = []

        for batch in train_loader:
            features = batch['features'].to(device)
            energies = batch['energy'].to(device)

            # Ensure energies has the right shape
            if energies.ndim == 0:
                energies = energies.unsqueeze(0)  # Add batch dimension if missing

            optimizer.zero_grad()
            predictions, _ = model(features)

            # Ensure predictions has the right shape
            if predictions.ndim == 0:
                predictions = predictions.unsqueeze(0)  # Add batch dimension if missing

            loss = F.mse_loss(predictions, energies)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            # Handle scalar and tensor outputs
            if predictions.ndim == 0:
                train_predictions.append(predictions.item())
            else:
                train_predictions.extend(predictions.detach().cpu().numpy())

            if energies.ndim == 0:
                train_targets.append(energies.item())
            else:
                train_targets.extend(energies.cpu().numpy())

        # Calculate training metrics
        train_loss /= len(train_loader)
        train_predictions = np.array(train_predictions)
        train_targets = np.array(train_targets)

        train_metrics['mse'].append(train_loss)
        train_metrics['mae'].append(np.mean(np.abs(train_predictions - train_targets)))
        train_metrics['r2'].append(r2_score(train_targets, train_predictions))

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                energies = batch['energy'].to(device)

                # Ensure energies has the right shape
                if energies.ndim == 0:
                    energies = energies.unsqueeze(0)

                predictions, _ = model(features)

                # Ensure predictions has the right shape
                if predictions.ndim == 0:
                    predictions = predictions.unsqueeze(0)

                loss = F.mse_loss(predictions, energies)

                val_loss += loss.item()

                # Handle scalar and tensor outputs
                if predictions.ndim == 0:
                    val_predictions.append(predictions.item())
                else:
                    val_predictions.extend(predictions.cpu().numpy())

                if energies.ndim == 0:
                    val_targets.append(energies.item())
                else:
                    val_targets.extend(energies.cpu().numpy())

        val_loss /= len(val_loader)
        val_predictions = np.array(val_predictions)
        val_targets = np.array(val_targets)

        val_metrics['mse'].append(val_loss)
        val_metrics['mae'].append(np.mean(np.abs(val_predictions - val_targets)))
        val_metrics['r2'].append(r2_score(val_targets, val_predictions))

        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"{epoch+1:5d} {train_loss:12.6f} {val_loss:12.6f} "
              f"{val_metrics['r2'][-1]:8.3f} {current_lr:10.2e}")

        # Save best model and check early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break

    if best_model is not None:
        model.load_state_dict(best_model)

    return model, train_metrics, val_metrics

def evaluate_model(model, test_loader, device, dataset):
    """Evaluate model performance with normalized values"""
    model.eval()
    predictions = []
    true_values = []
    
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device)
            energies = batch['energy']
            
            # Make predictions
            batch_predictions, _ = model(features)
            
            # Store predictions and true values (still normalized)
            predictions.extend(batch_predictions.cpu().numpy())
            true_values.extend(energies.cpu().numpy())
    
    # Convert to numpy arrays
    predictions = np.array(predictions)
    true_values = np.array(true_values)
    
    # Calculate metrics on normalized data
    rmse = np.sqrt(np.mean((predictions - true_values)**2))
    r2 = r2_score(true_values, predictions)
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Prediction accuracy (normalized) - MODIFIED TO SHOW R² VALUE
    plt.subplot(221)
    plt.scatter(true_values, predictions, alpha=0.5)
    plt.plot([true_values.min(), true_values.max()],
             [true_values.min(), true_values.max()],
             'r--', label=f'R² = {r2:.4f}')  # Changed label to show R² value
    plt.xlabel('True Energy (Normalized)')
    plt.ylabel('Predicted Energy (Normalized)')
    plt.title('Normalized Energy Prediction Accuracy')
    plt.grid(True)
    plt.legend()
    
    # Error distribution
    plt.subplot(222)
    errors = predictions - true_values
    sns.histplot(errors, bins=50)
    plt.xlabel('Error (Normalized Scale)')
    plt.ylabel('Count')
    plt.title('Prediction Error Distribution')
    
    # Residual plot
    plt.subplot(223)
    plt.scatter(predictions, errors, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Energy (Normalized)')
    plt.ylabel('Residual')
    plt.title('Residual Plot')
    plt.grid(True)
    
    # Histogram of true and predicted values
    plt.subplot(224)
    sns.histplot(true_values, bins=30, alpha=0.7, label='True Values')
    sns.histplot(predictions, bins=30, alpha=0.7, label='Predictions')
    plt.xlabel('Energy (Normalized)')
    plt.ylabel('Count')
    plt.title('Distribution of True and Predicted Values')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    print("\nModel Performance Metrics (Normalized Scale):")
    print("-" * 50)
    print(f"RMSE: {rmse:.6f}")
    print(f"R² Score: {r2:.6f}")
    
    return {
        'normalized_predictions': predictions,
        'normalized_true_values': true_values,
        'metrics': {
            'normalized_rmse': rmse,
            'normalized_r2': r2
        }
    }



def visualize_attenuation_effects(test_loader, dataset, device):
    """Analyze and visualize how attenuation affects signal characteristics"""
    # Group events by attenuation
    atten_groups = {}

    for batch in test_loader:
        features = batch['features'].cpu().numpy()
        energies = batch['energy'].numpy()

        for i in range(len(energies)):
            # Extract attenuation from features (last column)
            attens = features[i, :, 5]  # All antennas' attenuation values
            atten_bin = int(round(np.mean(attens)))

            if atten_bin not in atten_groups:
                atten_groups[atten_bin] = {
                    'features': [],
                    'energies': []
                }

            atten_groups[atten_bin]['features'].append(features[i])
            atten_groups[atten_bin]['energies'].append(energies[i])

    # Create visualization
    plt.figure(figsize=(15, 10))

    # Plot 1: Denormalized energy distribution by attenuation
    plt.subplot(221)
    legend_items = []  # Keep track of added items for legend
    
    for atten, data in sorted(atten_groups.items()):
        if len(data['energies']) < 5:
            continue

        energies = np.array(data['energies'])
        
        # Check if there's variance in the data
        if np.std(energies) > 1e-8:
            # Denormalize
            if hasattr(dataset, 'denormalize_energy'):
                energies = dataset.denormalize_energy(torch.tensor(energies)).numpy()
            
            # Use histplot instead of kdeplot for more robust visualization
            sns.histplot(energies, kde=True, stat="density", 
                         label=f'{atten} dB (n={len(energies)})')
            legend_items.append(f'{atten} dB (n={len(energies)})')
        else:
            print(f"Skipping KDE plot for attenuation {atten} dB due to zero variance")

    plt.xlabel('Source Energy')
    plt.ylabel('Density')
    plt.title('Energy Distribution by Attenuation')
    if legend_items:
        plt.legend()

    # Plot 2: Feature distributions (peak power) by attenuation
    plt.subplot(222)
    for atten, data in sorted(atten_groups.items()):
        if len(data['features']) < 5:
            continue

        # Get peak power feature (index 0)
        peak_powers = np.array([feat[:, 0].mean() for feat in data['features']])
        
        # Check for variance
        if np.std(peak_powers) > 1e-8:
            # Denormalize feature if normalization was applied
            if hasattr(dataset, 'feature_std') and hasattr(dataset, 'feature_mean'):
                # Using broadcasting to denormalize properly
                peak_powers = peak_powers * dataset.feature_std[0, 0, 0].item() + dataset.feature_mean[0, 0, 0].item()
            
            sns.histplot(peak_powers, kde=True, stat="density",
                         label=f'{atten} dB (n={len(peak_powers)})')

    plt.xlabel('Peak Power')
    plt.ylabel('Density')
    plt.title('Peak Power Distribution by Attenuation')
    plt.legend()

    # Plot 3: Attenuation vs Energy scatterplot
    plt.subplot(223)
    attens = []
    energies = []

    for atten, data in atten_groups.items():
        for e in data['energies']:
            attens.append(atten)
            energies.append(e)

    # Denormalize if needed
    if hasattr(dataset, 'denormalize_energy') and len(energies) > 0:
        energies = dataset.denormalize_energy(torch.tensor(energies)).numpy()

    plt.scatter(attens, energies, alpha=0.5)
    plt.xlabel('Attenuation (dB)')
    plt.ylabel('Source Energy')
    plt.title('Attenuation vs Energy')
    plt.grid(True)

    # Plot 4: Attenuation stats
    plt.subplot(224)
    atten_values = sorted(atten_groups.keys())
    counts = [len(atten_groups[a]['energies']) for a in atten_values]

    plt.bar(atten_values, counts)
    plt.xlabel('Attenuation (dB)')
    plt.ylabel('Count')
    plt.title('Events per Attenuation Setting')

    plt.tight_layout()
    plt.show()


def analyze_antenna_patterns(model, test_loader, device):
    """Analyze which antennas most influence the energy prediction"""
    model.eval()
    all_attention_maps = []

    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device)
            _, attention = model(features)
            all_attention_maps.extend(attention.cpu().numpy())

    attention_maps = np.array(all_attention_maps)

    # Compute average attention for each antenna
    mean_attention = attention_maps.mean(axis=0)

    # Create visualizations
    plt.figure(figsize=(15, 10))

    # Plot 1: Attention heatmap
    plt.subplot(221)
    im = plt.imshow(mean_attention.reshape(6, 8), cmap='viridis')
    plt.colorbar(im)
    plt.title('Average Antenna Attention')

    # Add antenna numbers
    for i in range(6):
        for j in range(8):
            ant_idx = i * 8 + j
            plt.text(j, i, str(ant_idx),
                    ha='center', va='center',
                    color='white' if mean_attention[ant_idx] > 0.03 else 'black',
                    fontsize=9)

    # Plot 2: Top 10 antennas
    plt.subplot(222)
    top_antennas = np.argsort(mean_attention)[-10:][::-1]
    top_values = mean_attention[top_antennas]

    plt.bar(range(len(top_antennas)), top_values)
    plt.xticks(range(len(top_antennas)), top_antennas)
    plt.xlabel('Antenna Index')
    plt.ylabel('Attention Weight')
    plt.title('Top 10 Antennas by Attention Weight')

    # Plot 3: Attention distribution
    plt.subplot(223)
    plt.hist(mean_attention, bins=30)
    plt.xlabel('Attention Weight')
    plt.ylabel('Count')
    plt.title('Attention Weight Distribution')

    # Plot 4: Attention variability across events
    plt.subplot(224)
    attention_std = attention_maps.std(axis=0)
    im = plt.imshow(attention_std.reshape(6, 8), cmap='plasma')
    plt.colorbar(im)
    plt.title('Attention Weight Variability (Std Dev)')

    plt.tight_layout()
    plt.show()

    return {
        'mean_attention': mean_attention,
        'attention_std': attention_std,
        'top_antennas': top_antennas
    }


def main():
    """Main execution function for enhanced WAIS energy prediction"""
    # Set up reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Configure computing device
    device = torch.device("cuda" if torch.cuda.is_available() else
                         "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define input files
    event_files =['INSERT_DESIRED_EVENT_FILES']
    header_files =['INSERT_DESIRED_HEADER_FILES']

    try:
        # Load and process WAIS calibration data
        print("\nLoading and processing WAIS calibration data with attenuation information...")
        features, source_energies, distances, event_times = get_wais_data(event_files, header_files)
        print(f"Processed {len(features)} WAIS calibration events")

        # Create dataset with attenuation awareness
        dataset = WAISDataset(features, source_energies, distances, event_times)

        # Create data splits
        train_dataset, val_dataset, test_dataset = create_improved_splits(
            dataset,
            train_size=0.7,
            val_size=0.15,
            seed=42
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True,
            pin_memory=True if device.type == "cuda" else False
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=32,
            pin_memory=True if device.type == "cuda" else False
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=32,
            pin_memory=True if device.type == "cuda" else False
        )

        # Visualize attenuation effects on signal features
        print("\nAnalyzing attenuation effects on signal characteristics...")
        visualize_attenuation_effects(train_loader, dataset, device)

        # Initialize enhanced model
        print("\nInitializing enhanced source energy prediction model...")
        model = EnhancedEnergyPredictor(input_features=features.shape[-1])
        model = model.to(device)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Train model
        print("\nTraining model with attenuation awareness...")
        model, train_metrics, val_metrics = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=100,
            device=device
        )

        # Evaluate model
        print("\nEvaluating model performance...")
        evaluation_results = evaluate_model(model, test_loader, device, dataset)

        # Analyze antenna attention patterns
        print("\nAnalyzing antenna attention patterns...")
        antenna_analysis = analyze_antenna_patterns(model, test_loader, device)

        # Save results
        save_dir = 'wais_energy_results'
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'evaluation_results': evaluation_results,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'antenna_analysis': antenna_analysis,
            'dataset_info': {
                'feature_mean': dataset.feature_mean.cpu().numpy(),
                'feature_std': dataset.feature_std.cpu().numpy(),
                'energy_mean': float(dataset.energy_mean),
                'energy_std': float(dataset.energy_std)
            }
        }, os.path.join(save_dir, f'enhanced_wais_model_{timestamp}.pt'))

        # Print final summary
        print("\nFinal Results Summary:")
        print("-" * 50)
        print(f"Model saved to: {save_dir}/enhanced_wais_model_{timestamp}.pt")
        print(f"Mean Relative Error: {evaluation_results['metrics']['mean_relative_error']:.2f}%")
        print(f"Median Relative Error: {evaluation_results['metrics']['median_relative_error']:.2f}%")
        print(f"R² Score: {evaluation_results['metrics']['r2']:.4f}")

        # Report top antennas
        print("\nTop 5 most important antennas:")
        for i, ant_idx in enumerate(antenna_analysis['top_antennas'][:5]):
            print(f"  #{i+1}: Antenna {ant_idx} (weight: {antenna_analysis['mean_attention'][ant_idx]:.4f})")

    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
