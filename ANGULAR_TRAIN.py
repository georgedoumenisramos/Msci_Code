import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pyproj import Transformer
import seaborn as sns
from scipy import interpolate
import torch.nn.functional as F
import os
from datetime import datetime


class AntGeom:
    def __init__(self):
        # Initialize transformers
        self.trans_GPS_to_XYZ = Transformer.from_crs(4979, 4978, always_xy=True)
        self.trans_GPS_to_APS = Transformer.from_crs(4979, 3031)

        # WAIS coordinates
        self.LATITUDE_WAIS_A4 = -79.468116
        self.LONGITUDE_WAIS_A4 = -112.059258
        self.ALTITUDE_WAIS_A4 = 1779.80

        # Calculate WAIS position
        self.waisX, self.waisY, self.waisZ = self.latlonaltToXYZ(
            self.LATITUDE_WAIS_A4,
            self.LONGITUDE_WAIS_A4,
            self.ALTITUDE_WAIS_A4
        )

        # Antenna mappings
        self.antToSurfMap = np.array([11, 5, 10, 4, 11, 4, 10, 5, 11, 5, 10, 4, 11, 4, 10, 5,
                                      9, 3, 8, 2, 8, 3, 9, 2, 9, 3, 8, 2, 8, 3, 9, 2,
                                      6, 0, 7, 1, 6, 1, 7, 0, 6, 0, 7, 1, 6, 1, 7, 0])

        self.vAntToChan = np.array([3, 1, 3, 5, 1, 3, 1, 3, 2, 0, 2, 0, 0, 2, 0, 2,
                                    1, 3, 1, 3, 3, 1, 3, 1, 0, 2, 0, 2, 2, 0, 2, 0,
                                    3, 1, 3, 1, 1, 3, 1, 3, 2, 0, 2, 0, 0, 2, 0, 2])

        self.hAntToChan = np.array([7, 5, 7, 1, 5, 7, 5, 7, 6, 4, 6, 4, 4, 6, 4, 6,
                                    5, 7, 5, 7, 7, 5, 7, 5, 4, 6, 4, 6, 6, 4, 6, 4,
                                    7, 5, 7, 5, 5, 7, 5, 7, 6, 4, 6, 4, 4, 6, 4, 6])

        self.surfChanToAnt = np.zeros((12, 9))
        for ant in range(len(self.antToSurfMap)):
            surf = self.antToSurfMap[ant]
            v = self.vAntToChan[ant]
            h = self.hAntToChan[ant]
            self.surfChanToAnt[surf][v] = -1 * (ant + 1)
            self.surfChanToAnt[surf][h] = +1 * (ant + 1)

    def latlonToAntarctica(self, lat, lon):
        return self.trans_GPS_to_APS.transform(lat, lon)

    def latlonaltToXYZ(self, lat, lon, alt):
        return self.trans_GPS_to_XYZ.transform(lon, lat, alt)

    def getAnglesToWais(self, latitudes, longitudes, altitudes, pitches, rolls, headings):
        """Calculate angles to WAIS in the balloon's reference frame."""
        # Get balloon position in XYZ
        balloon_x, balloon_y, balloon_z = self.latlonaltToXYZ(latitudes, longitudes, altitudes)

        # Vector to WAIS
        vector_x = self.waisX - balloon_x
        vector_y = self.waisY - balloon_y
        vector_z = self.waisZ - balloon_z

        return vector_x, vector_y, vector_z

    def getDistToWais(self, lat, lon, alt):
        """Calculate distance to WAIS."""
        x, y, z = self.latlonaltToXYZ(lat, lon, alt)
        return np.sqrt((x - self.waisX) ** 2 + (y - self.waisY) ** 2 + (z - self.waisZ) ** 2)
    def transform_to_local_frame(self, vector_x, vector_y, vector_z, pitch_deg, roll_deg, heading_deg):
        """Transform vectors from global frame to balloon's local frame using pitch, roll, and heading."""
        # Convert angles to radians
        heading_rad = np.radians(heading_deg)
        pitch_rad = np.radians(pitch_deg)
        roll_rad = np.radians(roll_deg)

        # Rotation matrices
        # Heading rotation (yaw), rotates around z-axis
        R_heading = np.array([
            [np.cos(heading_rad), np.sin(heading_rad), 0],
            [-np.sin(heading_rad), np.cos(heading_rad), 0],
            [0, 0, 1]
        ])

        # Pitch rotation, rotates around y-axis
        R_pitch = np.array([
            [np.cos(pitch_rad), 0, -np.sin(pitch_rad)],
            [0, 1, 0],
            [np.sin(pitch_rad), 0, np.cos(pitch_rad)]
        ])

        # Roll rotation, rotates around x-axis
        R_roll = np.array([
            [1, 0, 0],
            [0, np.cos(roll_rad), np.sin(roll_rad)],
            [0, -np.sin(roll_rad), np.cos(roll_rad)]
        ])

        # Combined rotation matrix from global to local frame
        R = (R_roll @ R_pitch @ R_heading).T  # Transpose to get inverse rotation

        # Apply rotation
        vector_global = np.array([vector_x, vector_y, vector_z])
        vector_local = R @ vector_global

        return vector_local[0], vector_local[1], vector_local[2]


def standardize_waveform(times, voltages, desired_start=-10, desired_end=110, num_points=248):
    """
    Standardize a waveform to have a specific time window and number of points

    Parameters:
        times: array of original time points
        voltages: array of original voltage values
        desired_start: start time in ns (default -10)
        desired_end: end time in ns (default 110)
        num_points: number of points in output (default 496 - doubled from original)
    """
    # Create new time array with desired range and increased number of points
    new_times = np.linspace(desired_start, desired_end, num_points)

    # Create interpolation function
    f = interpolate.interp1d(times, voltages, kind='linear',
                             bounds_error=False, fill_value=0.0)

    # Get new voltage values
    new_voltages = f(new_times)
    return new_times, new_voltages


def getTimesAndMillivolts(evgr, timeValues):
    """Get times and millivolts from event group."""
    adcOffset = np.array(evgr['data'].attrs['adcOffset'])
    mvScale = np.array(evgr['data'].attrs['mvScale'])
    timeOffset = np.array(evgr['data'].attrs['timeOffset'])
    timeScale = np.array(evgr['data'].attrs['timeScale'])
    validTimeInds = np.array(evgr['data'].attrs['validTimeInds'])
    chips = np.array(evgr['data'].attrs['chips'])
    data = np.array(evgr['data'])

    adcOffset = np.reshape(adcOffset, (adcOffset.shape[0], adcOffset.shape[1], -1))
    mvScale = np.reshape(mvScale, (mvScale.shape[0], mvScale.shape[1], -1))
    timeOffset = np.reshape(timeOffset, (timeOffset.shape[0], timeOffset.shape[1], -1))
    timeScale = np.reshape(timeScale, (timeScale.shape[0], timeScale.shape[1], -1))

    data = mvScale * (data - adcOffset)
    times = np.zeros((12, 9, 250))
    N = np.zeros((12, 9), dtype=int)

    for surf in range(12):
        for chan in range(9):
            times[surf][chan][0:validTimeInds[surf][1] - validTimeInds[surf][0]] = \
                timeValues[surf][chips[surf]][validTimeInds[surf][0]:validTimeInds[surf][1]]
            times[surf][chan][:] = times[surf][chan][:] - times[surf][chan][0]
            N[surf][chan] = validTimeInds[surf][1] - validTimeInds[surf][0]

    times = (times * timeScale) + timeOffset
    return N, times, data


def process_waveform_for_ml(event, geom, timeValues):
    """Process a single event's waveforms for ML input."""
    data = np.array(event['data'])
    N, times, voltages = getTimesAndMillivolts(event, timeValues)

    # Initialize standardized voltage image
    v_image = np.zeros((48, 248))  # Changed to 248 points

    for ant in range(48):
        surf = geom.antToSurfMap[ant]
        chan = geom.vAntToChan[ant]
        if surf < voltages.shape[0] and chan < voltages.shape[1]:
            orig_times = times[surf][chan][:N[surf][chan]][:-2]
            orig_voltages = voltages[surf][chan][:N[surf][chan]][:-2]

            _, new_voltages = standardize_waveform(orig_times, orig_voltages)
            v_image[ant] = new_voltages

    return v_image



class WAISDataset(Dataset):
    def __init__(self, voltage_data, angles, pitches, rolls, headings):
        self.voltage_data = torch.FloatTensor(voltage_data)
        self.angles = torch.FloatTensor(angles)
        self.pitches = torch.FloatTensor(pitches)
        self.rolls = torch.FloatTensor(rolls)
        self.headings = torch.FloatTensor(headings)

        # Update input validation for new dimensions
        assert self.voltage_data.shape[1:] == (48, 248), \
            f"Expected voltage shape (N, 48, 248), got {self.voltage_data.shape}"
        assert self.angles.shape[1] == 3, \
            f"Expected angles shape (N, 3), got {self.angles.shape}"
        assert len(self.voltage_data) == len(self.angles) == len(self.pitches) == len(self.rolls) == len(self.headings), \
            "Mismatch between data lengths"

    def __len__(self):
        return len(self.voltage_data)

    def __getitem__(self, idx):
        voltage = self.voltage_data[idx].unsqueeze(0)  # Add channel dimension
        angle = self.angles[idx]
        pitch = self.pitches[idx]
        roll = self.rolls[idx]
        heading = self.headings[idx]
        return voltage, angle, pitch, roll, heading

def normalize_voltage_data(voltage_data):
    """Normalize each voltage image individually."""
    assert isinstance(voltage_data, np.ndarray), "Expected numpy array"
    assert voltage_data.ndim == 3, f"Expected 3D array, got shape {voltage_data.shape}"

    normalized = np.zeros_like(voltage_data, dtype=np.float64)

    for i in range(len(voltage_data)):
        image = voltage_data[i]
        # Check for constant images
        if np.std(image) < 1e-14:
            print(f"Warning: Nearly constant voltage data in event {i}")
            continue

        # Normalize to [-1, 1]
        min_val = np.min(image)
        max_val = np.max(image)
        normalized[i] = 2 * (image - min_val) / (max_val - min_val + 1e-12) - 1

        # Validate normalization
        assert -1.01 <= normalized[i].min() <= 1.01, f"Normalization error: min value {normalized[i].min()}"
        assert -1.01 <= normalized[i].max() <= 1.01, f"Normalization error: max value {normalized[i].max()}"

    return normalized

def analyze_wais_angles_vs_voltage(header_file, event_file, model, device='mps'):
    """
    Analyze relationship between predicted and true azimuthal angles for WAIS events.

    Args:
        header_file: Path to header HDF5 file
        event_file: Path to event HDF5 file
        model: Trained WAIS CNN model
        device: Computing device (default='mps')
    """
    try:
        run_number = event_file.split('event')[-1].split('.')[0]
        run_key = f'run{run_number}'
        print(f"Analyzing {run_key}")

        with h5py.File(header_file, 'r') as hf, h5py.File(event_file, 'r') as ef:
            geom = AntGeom()
            run_header = hf[run_key]
            run_events = ef[run_key]
            calgr = ef['calib']
            timeValues = np.array(calgr['timeValues'])

            is_wais = run_header['isWAIS'][:]
            wais_indices = np.where(is_wais)[0]
            print(f"Found {len(wais_indices)} WAIS events")

            # Lists to store results
            true_azimuths = []
            predicted_azimuths = []
            max_voltage_antennas = []

            # Process each WAIS event
            for i, idx in enumerate(wais_indices):
                event_num = run_header['eventNumber'][idx]
                ev_name = f'ev_{event_num}'

                if ev_name not in run_events:
                    continue

                event = run_events[ev_name]

                # Get position data
                lat = run_header['latitude'][idx]
                lon = run_header['longitude'][idx]
                alt = run_header['altitude'][idx]
                pitch = run_header['pitch'][idx]
                roll = run_header['roll'][idx]
                heading = run_header['heading'][idx]

                # Calculate true WAIS vector in global frame
                x_global, y_global, z_global = geom.getAnglesToWais(
                    np.array([lat]), np.array([lon]), np.array([alt]),
                    np.array([0]), np.array([0]), np.array([0])
                )
                x_global = x_global[0]
                y_global = y_global[0]
                z_global = z_global[0]

                # Transform true vector to local frame
                x_local_true, y_local_true, z_local_true = geom.transform_to_local_frame(
                    x_global, y_global, z_global, pitch, roll, heading
                )

                # Compute true azimuth in local frame
                true_azimuth = np.degrees(np.arctan2(y_local_true, x_local_true)) % 360

                # Process voltage data
                v_image = process_waveform_for_ml(event, geom, timeValues)


                # Get peak-to-peak voltage for each antenna
                voltage_pp = np.ptp(v_image, axis=1)
                max_voltage_ant = np.argmax(voltage_pp)

                # Prepare input for model
                # First add batch dimension
                normalized_voltage = normalize_voltage_data(v_image[np.newaxis, :, :])

                # Convert to tensor and add channel dimension
                voltage_tensor = torch.FloatTensor(normalized_voltage).unsqueeze(1)

                # Move to device
                voltage_tensor = voltage_tensor.to(device)

                # Get model prediction
                with torch.no_grad():
                    pred_vec_global = model(voltage_tensor).cpu().numpy()[0]
                    # Ensure the predicted vector is normalized
                    pred_vec_global /= np.linalg.norm(pred_vec_global) + 1e-10

                # Transform predicted vector to local frame
                x_local_pred, y_local_pred, z_local_pred = geom.transform_to_local_frame(
                    pred_vec_global[0], pred_vec_global[1], pred_vec_global[2],
                    pitch, roll, heading
                )

                # Compute predicted azimuth in local frame
                pred_azimuth = np.degrees(np.arctan2(y_local_pred, x_local_pred)) % 360

                # Store results
                true_azimuths.append(true_azimuth)
                predicted_azimuths.append(pred_azimuth)
                max_voltage_antennas.append(max_voltage_ant)

                # Print progress every 10 events
                if i % 20 == 0:
                    print(f"Processed {i+1}/{len(wais_indices)} events")

            # Convert to arrays
            true_azimuths = np.array(true_azimuths)
            predicted_azimuths = np.array(predicted_azimuths)
            max_voltage_antennas = np.array(max_voltage_antennas)

            # Calculate error metrics
            angle_errors = np.abs((predicted_azimuths - true_azimuths + 180) % 360 - 180)
            mean_error = np.mean(angle_errors)
            median_error = np.median(angle_errors)
            correlation = np.corrcoef(true_azimuths, predicted_azimuths)[0, 1]

            print("\nAnalysis Results:")
            print(f"Correlation with predicted angles: {correlation:.3f}")
            print(f"Mean prediction error: {mean_error:.2f}°")
            print(f"Median prediction error: {median_error:.2f}°")

            # Create visualization
            plt.figure(figsize=(15, 8))
            plt.scatter(true_azimuths, max_voltage_antennas,
                       alpha=0.6, color='red', marker='o', label='True', s=50)
            plt.scatter(predicted_azimuths, max_voltage_antennas,
                       alpha=0.6, color='blue', marker='o', label='Predicted', s=50)
            plt.xlabel('Azimuthal Angle (degrees)')
            plt.ylabel('Antenna Number')
            plt.title(f'Maximum Voltage Antenna vs Azimuthal Angle - {run_key}')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.show()

            return true_azimuths, predicted_azimuths, max_voltage_antennas, angle_errors

    except Exception as e:
        print(f"Error analyzing {event_file}: {str(e)}")
        raise

def load_paired_wais_data(event_files, header_files):
    """Load and process paired WAISH-WAIS data using adjacent event numbers."""
    raw_voltage_data_h = [] # WAISH events
    raw_voltage_data_v = [] # WAIS events
    all_angles = []
    all_pitches = []
    all_rolls = []
    all_headings = []

    for event_file, header_file in zip(event_files, header_files):
        run_key = f'run{event_file.split("event")[-1].split(".")[0]}'

        with h5py.File(event_file, 'r') as ef, h5py.File(header_file, 'r') as hf:
            geom = AntGeom()
            run_header = hf[run_key]
            run_events = ef[run_key]
            timeValues = np.array(ef['calib']['timeValues'])

            # Get event flags and numbers
            event_numbers = run_header['eventNumber'][:]
            is_wais = run_header['isWAIS'][:]
            is_waish = run_header['isWAISH'][:]

            # Find WAIS and WAISH events
            wais_indices = np.where(is_wais)[0]
            waish_indices = np.where(is_waish)[0]

            print(f"\nFound {len(wais_indices)} WAIS and {len(waish_indices)} WAISH events")

            # Match events by adjacent event numbers
            for i in range(len(waish_indices)):
                waish_idx = waish_indices[i]
                waish_event_num = event_numbers[waish_idx]

                # Look for matching WAIS event with next event number
                wais_matches = np.where(event_numbers == waish_event_num + 1)[0]

                if len(wais_matches) > 0 and wais_matches[0] in wais_indices:
                    wais_idx = wais_matches[0]

                    # Get event names
                    ev_name_h = f'ev_{event_numbers[waish_idx]}'
                    ev_name_v = f'ev_{event_numbers[wais_idx]}'

                    if ev_name_h in run_events and ev_name_v in run_events:
                        try:
                            # Process both waveforms
                            v_image_h = process_waveform_for_ml(run_events[ev_name_h], geom, timeValues)
                            v_image_v = process_waveform_for_ml(run_events[ev_name_v], geom, timeValues)

                            # Get angles using WAIS event timing
                            vector = geom.getAnglesToWais(
                                np.array([run_header['latitude'][wais_idx]]),
                                np.array([run_header['longitude'][wais_idx]]),
                                np.array([run_header['altitude'][wais_idx]]),
                                np.array([run_header['pitch'][wais_idx]]),
                                np.array([run_header['roll'][wais_idx]]),
                                np.array([run_header['heading'][wais_idx]])
                            )

                            raw_voltage_data_h.append(v_image_h)
                            raw_voltage_data_v.append(v_image_v)
                            all_angles.append([vector[0][0], vector[1][0], vector[2][0]])
                            all_pitches.append(run_header['pitch'][wais_idx])
                            all_rolls.append(run_header['roll'][wais_idx])
                            all_headings.append(run_header['heading'][wais_idx])

                        except Exception as e:
                            print(f"Error processing pair ({ev_name_h}, {ev_name_v}): {str(e)}")
                            continue

    # Convert to arrays
    voltage_data_h = np.array(raw_voltage_data_h)
    voltage_data_v = np.array(raw_voltage_data_v)
    angles = np.array(all_angles)
    pitches = np.array(all_pitches)
    rolls = np.array(all_rolls)
    headings = np.array(all_headings)

    print(f"\nSuccessfully paired {len(voltage_data_h)} WAIS-WAISH events")
    print(f"Voltage data shapes: WAISH {voltage_data_h.shape}, WAIS {voltage_data_v.shape}")

    return voltage_data_h, voltage_data_v, angles, pitches, rolls, headings

class PairedWAISDataset(Dataset):
    def __init__(self, voltage_data_h, voltage_data_v, angles, pitches, rolls, headings):
        self.voltage_data_h = torch.FloatTensor(voltage_data_h)
        self.voltage_data_v = torch.FloatTensor(voltage_data_v)
        self.angles = torch.FloatTensor(angles)
        self.pitches = torch.FloatTensor(pitches)
        self.rolls = torch.FloatTensor(rolls)
        self.headings = torch.FloatTensor(headings)

    def __len__(self):
        return len(self.voltage_data_h)

    def __getitem__(self, idx):
        voltage_h = self.voltage_data_h[idx].unsqueeze(0)
        voltage_v = self.voltage_data_v[idx].unsqueeze(0)
        angle = self.angles[idx]
        pitch = self.pitches[idx]
        roll = self.rolls[idx]
        heading = self.headings[idx]
        return voltage_h, voltage_v, angle, pitch, roll, heading


def compute_angular_error(pred_angles, true_angles):
    """
    Compute the angular error accounting for periodicity.
    """
    # Convert to radians
    pred_azimuth = np.radians(pred_angles[:, 0])
    true_azimuth = np.radians(true_angles[:, 0])

    # Compute the minimum angular difference for azimuth
    azimuth_diff = np.abs(np.arctan2(
        np.sin(pred_azimuth - true_azimuth),
        np.cos(pred_azimuth - true_azimuth)
    ))
    azimuth_error = np.degrees(azimuth_diff)

    # Regular difference for elevation
    elevation_error = np.abs(pred_angles[:, 1] - true_angles[:, 1])

    return azimuth_error, elevation_error


def train_model(model, train_loader, val_loader, num_epochs=50):
    """Train the model with validation."""
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    def cart_loss(output, target):
        """
        Normalized Cartesian coordinate loss function.
        Normalizes both output and target vectors before computing loss.
        """
        # Normalize the vectors
        output_norm = F.normalize(output, p=2, dim=1)
        target_norm = F.normalize(target, p=2, dim=1)

        # Compute MSE loss between normalized vectors
        loss = torch.mean((output_norm - target_norm)**2)

        return loss

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for voltage, xyz, _, _, _ in train_loader:  # We don't need pitch, roll, heading during training
            voltage, xyz = voltage.to(device), xyz.to(device)
            optimizer.zero_grad()
            output = model(voltage)
            loss = cart_loss(output, xyz)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for voltage, xyz, pitch, roll, heading in val_loader:
                voltage, xyz = voltage.to(device), xyz.to(device)
                output = model(voltage)
                val_loss += cart_loss(output, xyz).item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Print progress
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.8f}")
        print(f"Val Loss: {val_loss:.8f}")

        # Evaluate angles every N epochs
        if (epoch + 1) % 20 == 0:
            print("\nEvaluating angle predictions...")
            stats = evaluate_model(model, val_loader, device)
            print(f"Validation Mean Azimuth Error: {stats['mean_azimuth_error']:.2f}°")
            print(f"Validation Median Azimuth Error: {stats['median_azimuth_error']:.2f}°")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')

    return train_losses, val_losses

class EnhancedWAISCNN(nn.Module):
    """Enhanced CNN for WAIS angle prediction using both polarizations as channels."""
    def __init__(self):
        super().__init__()

        # Modified conv layers to accept 2 input channels (WAISH and WAIS)
        self.conv_layers = nn.Sequential(
            # First layer takes 2 channels (WAISH and WAIS)
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self._calc_flat_features = self._get_flat_features()

        self.fc_layers = nn.Sequential(
            nn.Linear(self._calc_flat_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.cartesian_head = nn.Linear(256, 3)

    def _get_flat_features(self):
        # Helper function to calculate the flat features size
        x = torch.randn(1, 2, 48, 248)
        x = self.conv_layers(x)
        return x.numel() // x.size(0)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, self._calc_flat_features)
        x = self.fc_layers(x)
        x = self.cartesian_head(x)
        x = F.normalize(x, p=2, dim=1)
        return x
def train_dual_polarization_model(model, train_loader, val_loader, num_epochs=50):
    """Train the model with simple normalized Cartesian loss."""
    device = torch.device("mps" if torch.backends.mps.is_available() else
                         "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    train_losses = []
    val_losses = []

    def normalized_cart_loss(output, target):
        """Simple normalized Cartesian coordinate loss."""
        # Normalize the vectors
        output_norm = F.normalize(output, p=2, dim=1)
        target_norm = F.normalize(target, p=2, dim=1)

        # Compute MSE loss between normalized vectors
        loss = torch.mean((output_norm - target_norm)**2)
        return loss

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        batch_count = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')

        for voltage_h, voltage_v, xyz, pitch, roll, heading in progress_bar:
            voltage_combined = torch.stack([voltage_h.squeeze(1), voltage_v.squeeze(1)], dim=1)
            voltage_combined, xyz = voltage_combined.to(device), xyz.to(device)

            voltage_combined = (voltage_combined - voltage_combined.mean()) / (voltage_combined.std() + 1e-8)

            optimizer.zero_grad()
            output = model(voltage_combined)
            loss = normalized_cart_loss(output, xyz)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            batch_count += 1

            progress_bar.set_postfix({
                'training_loss': f'{loss.item():.6f}',
                'avg_loss': f'{(train_loss/batch_count):.6f}'
            })

        train_loss /= batch_count
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0
        val_batch_count = 0
        all_pred_xyz = []
        all_true_xyz = []
        all_pitches = []
        all_rolls = []
        all_headings = []

        with torch.no_grad():
            for voltage_h, voltage_v, xyz, pitch, roll, heading in val_loader:
                voltage_combined = torch.stack([voltage_h.squeeze(1), voltage_v.squeeze(1)], dim=1)
                voltage_combined, xyz = voltage_combined.to(device), xyz.to(device)

                voltage_combined = (voltage_combined - voltage_combined.mean()) / (voltage_combined.std() + 1e-8)

                output = model(voltage_combined)
                val_loss += normalized_cart_loss(output, xyz).item()
                val_batch_count += 1

                all_pred_xyz.append(output.cpu().numpy())
                all_true_xyz.append(xyz.cpu().numpy())
                all_pitches.append(pitch.numpy())
                all_rolls.append(roll.numpy())
                all_headings.append(heading.numpy())

        val_loss /= val_batch_count
        val_losses.append(val_loss)

        # Combine validation results
        all_pred_xyz = np.concatenate(all_pred_xyz)
        all_true_xyz = np.concatenate(all_true_xyz)
        all_pitches = np.concatenate(all_pitches)
        all_rolls = np.concatenate(all_rolls)
        all_headings = np.concatenate(all_headings)

        # Calculate validation statistics
        stats = compare_vector_angles(
            all_pred_xyz, all_true_xyz,
            all_pitches, all_rolls, all_headings
        )

        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Training Loss: {train_loss:.6f}")
        print(f"Validation Loss: {val_loss:.6f}")
        print(f"Mean Azimuth Error: {stats['mean_azimuth_error']:.2f}°")
        print(f"Mean Elevation Error: {stats['mean_elevation_error']:.2f}°")
        print(f"Current Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")

        # Learning rate scheduling
        scheduler.step(val_loss)

    return model, train_losses, val_losses

def verify_training_data(angles, normalized_angles, denormalized_angles):
    """Verify that normalization/denormalization preserves angle relationships."""
    print("\nVerifying angle transformations:")
    print("Original angles shape:", angles.shape)
    print("Normalized angles shape:", normalized_angles.shape)
    print("Denormalized angles shape:", denormalized_angles.shape)

    # Check ranges
    print("\nAngle ranges:")
    print(f"Original azimuth: [{angles[:, 0].min():.3f}, {angles[:, 0].max():.3f}]")
    print(f"Original elevation: [{angles[:, 1].min():.3f}, {angles[:, 1].max():.3f}]")

    # For normalized angles, check sin/cos components
    print(f"Normalized sin(azimuth): [{normalized_angles[:, 0].min():.3f}, {normalized_angles[:, 0].max():.3f}]")
    print(f"Normalized cos(azimuth): [{normalized_angles[:, 1].min():.3f}, {normalized_angles[:, 1].max():.3f}]")
    print(f"Normalized elevation: [{normalized_angles[:, 2].min():.3f}, {normalized_angles[:, 2].max():.3f}]")

    print(f"Denormalized azimuth: [{denormalized_angles[:, 0].min():.3f}, {denormalized_angles[:, 0].max():.3f}]")
    print(f"Denormalized elevation: [{denormalized_angles[:, 1].min():.3f}, {denormalized_angles[:, 1].max():.3f}]")

    # Check reconstruction error using angular difference
    azimuth_error, elevation_error = compute_angular_error(denormalized_angles, angles)
    print("\nReconstruction errors:")
    print(f"Max azimuth error: {np.max(azimuth_error):.6f}°")
    print(f"Max elevation error: {np.max(elevation_error):.6f}°")

def plot_training_history(train_losses, val_losses):
    """Plot training history."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.yscale('log')
    plt.grid(True)
    plt.show()

def normalize_vectors(vectors):
    """Normalize XYZ vectors to unit vectors"""
    # Remove extra dimension if present
    if vectors.ndim == 3:
        vectors = vectors.squeeze(-1)

    # Calculate magnitude of each vector
    magnitudes = np.sqrt(np.sum(vectors**2, axis=1, keepdims=True))
    # Avoid division by zero
    magnitudes = np.where(magnitudes == 0, 1e-10, magnitudes)
    # Normalize vectors
    normalized = vectors / magnitudes
    return normalized


def load_wais_data(event_files, header_files):
    """Load and process WAIS data from multiple run files."""

    def process_event_angles(geom, lat, lon, alt, pitch, roll, heading):
        """Calculate normalized vector to WAIS."""
        x, y, z = geom.getAnglesToWais(
            np.array([lat]), np.array([lon]), np.array([alt]),
            np.array([pitch]), np.array([roll]), np.array([heading])
        )
        vector = np.array([x[0], y[0], z[0]])
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector

    # Lists to store all data
    raw_voltage_data = []
    total_events = 0
    num_wais_events = 0
    all_pitches = []
    all_rolls = []
    all_headings = []

    # First pass: collect all voltage data to find global maximum
    print("\nFirst pass: collecting voltage data...")
    for event_file, header_file in zip(event_files, header_files):
        # Extract run number from filename
        run_number = event_file.split('event')[-1].split('.')[0]
        run_key = f'run{run_number}'
        print(f"\nProcessing {run_key} from {event_file}")

        try:
            with h5py.File(event_file, 'r') as ef, h5py.File(header_file, 'r') as hf:
                if run_key not in hf:
                    print(f"Warning: {run_key} not found in {header_file}")
                    continue

                geom = AntGeom()
                run_header = hf[run_key]
                run_events = ef[run_key]
                calgr = ef['calib']
                timeValues = np.array(calgr['timeValues'])

                is_wais = run_header['isWAIS'][:]
                wais_indices = np.where(is_wais)[0]
                num_wais_events += len(wais_indices)

                print(f"Found {len(wais_indices)} WAIS events in {run_key}")

                for i, idx in enumerate(wais_indices):
                    if i % 100 == 0:
                        print(f"Processing event {i}/{len(wais_indices)}")

                    event_num = run_header['eventNumber'][idx]
                    ev_name = f'ev_{event_num}'

                    if ev_name not in run_events:
                        continue

                    try:
                        event = run_events[ev_name]
                        v_image = process_waveform_for_ml(event, geom, timeValues)
                        raw_voltage_data.append(v_image)

                    except Exception as e:
                        print(f"Error processing event {ev_name}: {str(e)}")
                        continue

        except Exception as e:
            print(f"Error processing {event_file}: {str(e)}")
            continue

    if not raw_voltage_data:
        raise ValueError("No valid data found in any of the input files")

    # Convert to numpy array and find global maximum
    raw_voltage_data = np.array(raw_voltage_data)
    global_max = np.max(np.abs(raw_voltage_data))
    print(f"\nGlobal maximum voltage: {global_max:.2f}")

    # Second pass to normalize with global maximum and collect angles
    print("\nSecond pass: normalizing data and collecting angles...")
    all_voltage_data = []
    all_angles = []
    all_wais_indices = []

    for event_file, header_file in zip(event_files, header_files):
        run_number = event_file.split('event')[-1].split('.')[0]
        run_key = f'run{run_number}'

        try:
            with h5py.File(event_file, 'r') as ef, h5py.File(header_file, 'r') as hf:
                if run_key not in hf:
                    continue

                geom = AntGeom()
                run_header = hf[run_key]
                run_events = ef[run_key]
                calgr = ef['calib']
                timeValues = np.array(calgr['timeValues'])

                is_wais = run_header['isWAIS'][:]
                wais_indices = np.where(is_wais)[0]

                file_voltage_data = []
                file_angles = []
                file_pitches = []
                file_rolls = []
                file_headings = []

                for i, idx in enumerate(wais_indices):
                    event_num = run_header['eventNumber'][idx]
                    ev_name = f'ev_{event_num}'

                    if ev_name not in run_events:
                        continue

                    try:
                        event = run_events[ev_name]
                        v_image = process_waveform_for_ml(event, geom, timeValues)
                        v_image = v_image / global_max  # Normalize using global maximum

                        vector = process_event_angles(
                            geom,
                            run_header['latitude'][idx],
                            run_header['longitude'][idx],
                            run_header['altitude'][idx],
                            run_header['pitch'][idx],
                            run_header['roll'][idx],
                            run_header['heading'][idx]
                        )

                        file_angles.append(vector)
                        file_voltage_data.append(v_image)
                        file_pitches.append(run_header['pitch'][idx])
                        file_rolls.append(run_header['roll'][idx])
                        file_headings.append(run_header['heading'][idx])

                    except Exception as e:
                        print(f"Error processing event {ev_name}: {str(e)}")
                        continue

                if len(file_voltage_data) > 0:
                    file_voltage_data = np.array(file_voltage_data)
                    file_angles = np.array(file_angles)
                    file_pitches = np.array(file_pitches)
                    file_rolls = np.array(file_rolls)
                    file_headings = np.array(file_headings)
                    all_voltage_data.append(file_voltage_data)
                    all_angles.append(file_angles)
                    all_pitches.append(file_pitches)
                    all_rolls.append(file_rolls)
                    all_headings.append(file_headings)
                    all_wais_indices.append(wais_indices + total_events)
                    total_events += len(is_wais)

        except Exception as e:
            print(f"Error processing {event_file}: {str(e)}")
            continue

    # Combine all data
    voltage_data = np.concatenate(all_voltage_data, axis=0)
    angles = np.concatenate(all_angles, axis=0)
    pitches = np.concatenate(all_pitches, axis=0)
    rolls = np.concatenate(all_rolls, axis=0)
    headings = np.concatenate(all_headings, axis=0)
    wais_indices = np.concatenate(all_wais_indices)

    print("\nData processing completed:")
    print(f"Total voltage data shape: {voltage_data.shape}")
    print(f"Total angles shape: {angles.shape}")
    print(f"Total pitches shape: {pitches.shape}")
    print(f"Total rolls shape: {rolls.shape}")
    print(f"Total headings shape: {headings.shape}")
    print(f"Total WAIS events processed: {len(wais_indices)}")

    return voltage_data, angles, pitches, rolls, headings, wais_indices

def compare_vector_angles(predicted_xyz, true_xyz, pitches, rolls, headings):
    """
    Convert XYZ vectors to azimuth and elevation angles in the balloon's local frame and compare them.

    Args:
        predicted_xyz: numpy array of shape (N, 3) containing predicted XYZ coordinates (global frame)
        true_xyz: numpy array of shape (N, 3) containing true XYZ coordinates (global frame)
        pitches: numpy array of shape (N,) containing pitch angles in degrees
        rolls: numpy array of shape (N,) containing roll angles in degrees
        headings: numpy array of shape (N,) containing heading angles in degrees

    Returns:
        Dictionary containing angle differences and statistics
    """

    # Instantiate AntGeom to use transform_to_local_frame method
    geom = AntGeom()

    # Initialize lists to store azimuth and elevation angles
    pred_azimuths = []
    pred_elevations = []
    true_azimuths = []
    true_elevations = []

    # Loop over each sample to transform vectors and compute angles
    for i in range(len(predicted_xyz)):
        # Get predicted and true vectors (global frame)
        pred_vec_global = predicted_xyz[i]
        true_vec_global = true_xyz[i]

        # Normalize vectors
        pred_vec_global /= np.linalg.norm(pred_vec_global) + 1e-10
        true_vec_global /= np.linalg.norm(true_vec_global) + 1e-10

        # Extract pitch, roll, heading for this sample
        pitch = pitches[i]
        roll = rolls[i]
        heading = headings[i]

        # Transform predicted vector to local frame
        x_local_pred, y_local_pred, z_local_pred = geom.transform_to_local_frame(
            pred_vec_global[0], pred_vec_global[1], pred_vec_global[2],
            pitch, roll, heading
        )

        # Transform true vector to local frame
        x_local_true, y_local_true, z_local_true = geom.transform_to_local_frame(
            true_vec_global[0], true_vec_global[1], true_vec_global[2],
            pitch, roll, heading
        )

        # Normalize the transformed vectors
        mag_pred = np.linalg.norm([x_local_pred, y_local_pred, z_local_pred]) + 1e-10
        x_local_pred_norm = x_local_pred / mag_pred
        y_local_pred_norm = y_local_pred / mag_pred
        z_local_pred_norm = z_local_pred / mag_pred

        mag_true = np.linalg.norm([x_local_true, y_local_true, z_local_true]) + 1e-10
        x_local_true_norm = x_local_true / mag_true
        y_local_true_norm = y_local_true / mag_true
        z_local_true_norm = z_local_true / mag_true

        # Compute azimuth angles in local frame
        pred_azimuth = np.degrees(np.arctan2(y_local_pred_norm, x_local_pred_norm)) % 360
        true_azimuth = np.degrees(np.arctan2(y_local_true_norm, x_local_true_norm)) % 360

        # Compute elevation angles in local frame
        pred_elevation = np.degrees(np.arcsin(z_local_pred_norm))
        true_elevation = np.degrees(np.arcsin(z_local_true_norm))

        # Append to lists
        pred_azimuths.append(pred_azimuth)
        pred_elevations.append(pred_elevation)
        true_azimuths.append(true_azimuth)
        true_elevations.append(true_elevation)

    # Convert lists to numpy arrays
    pred_azimuths = np.array(pred_azimuths)
    pred_elevations = np.array(pred_elevations)
    true_azimuths = np.array(true_azimuths)
    true_elevations = np.array(true_elevations)

    # Calculate angle differences
    # Handle azimuth wraparound by taking the smallest angle difference
    azimuth_diff = np.abs((pred_azimuths - true_azimuths + 180) % 360 - 180)
    elevation_diff = np.abs(pred_elevations - true_elevations)

    # Calculate statistics
    stats = {
        'mean_azimuth_error': np.mean(azimuth_diff),
        'median_azimuth_error': np.median(azimuth_diff),
        'std_azimuth_error': np.std(azimuth_diff),
        'mean_elevation_error': np.mean(elevation_diff),
        'median_elevation_error': np.median(elevation_diff),
        'std_elevation_error': np.std(elevation_diff),
        'pred_azimuths': pred_azimuths,
        'pred_elevations': pred_elevations,
        'true_azimuths': true_azimuths,
        'true_elevations': true_elevations,
        'azimuth_diff': azimuth_diff,
        'elevation_diff': elevation_diff
    }


    # Create visualization
    plt.figure(figsize=(15, 10))

    # Plot azimuth comparison
    plt.subplot(221)
    plt.scatter(true_azimuths, pred_azimuths, alpha=0.5)
    plt.plot([0, 360], [0, 360], 'r--')
    plt.xlabel('True Azimuth (degrees)')
    plt.ylabel('Predicted Azimuth (degrees)')
    plt.title('Azimuth Comparison')
    plt.grid(True)

    # Plot elevation comparison
    plt.subplot(222)
    plt.scatter(true_elevations, pred_elevations, alpha=0.5)
    plt.plot([-90, 90], [-90, 90], 'r--')
    plt.xlabel('True Elevation (degrees)')
    plt.ylabel('Predicted Elevation (degrees)')
    plt.title('Elevation Comparison')
    plt.grid(True)

    # Plot angle errors histogram
    plt.subplot(223)
    plt.hist(azimuth_diff, bins=50, alpha=0.7, label='Azimuth')
    plt.xlabel('Error (degrees)')
    plt.ylabel('Count')
    plt.title(f'Azimuth Error Distribution\nMean: {stats["mean_azimuth_error"]:.3f}°')
    plt.grid(True)

    plt.subplot(224)
    plt.hist(elevation_diff, bins=50, alpha=0.7, label='Elevation')
    plt.xlabel('Error (degrees)')
    plt.ylabel('Count')
    plt.title(f'Elevation Error Distribution\nMean: {stats["mean_elevation_error"]:.3f}°')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Print statistics
    print("\nAngle Prediction Statistics:")
    print("=" * 50)
    print(f"Azimuth Error:")
    print(f"  Mean: {stats['mean_azimuth_error']:.3f}°")
    print(f"  Median: {stats['median_azimuth_error']:.3f}°")
    print(f"  Std Dev: {stats['std_azimuth_error']:.3f}°")
    print("\nElevation Error:")
    print(f"  Mean: {stats['mean_elevation_error']:.3f}°")
    print(f"  Median: {stats['median_elevation_error']:.3f}°")
    print(f"  Std Dev: {stats['std_elevation_error']:.3f}°")

    return stats

def evaluate_model(model, test_loader, device):
    model.eval()
    all_pred_xyz = []
    all_true_xyz = []
    all_pitches = []
    all_rolls = []
    all_headings = []

    with torch.no_grad():
        for voltage, true_xyz, pitch, roll, heading in test_loader:
            voltage = voltage.to(device)
            output = model(voltage)
            all_pred_xyz.append(output.cpu().numpy())
            all_true_xyz.append(true_xyz.numpy())
            all_pitches.append(pitch.numpy())
            all_rolls.append(roll.numpy())
            all_headings.append(heading.numpy())

    # Concatenate arrays
    all_pred_xyz = np.concatenate(all_pred_xyz, axis=0)
    all_true_xyz = np.concatenate(all_true_xyz, axis=0)
    pitches = np.concatenate(all_pitches, axis=0)
    rolls = np.concatenate(all_rolls, axis=0)
    headings = np.concatenate(all_headings, axis=0)

    # Call compare_vector_angles with orientation data
    stats = compare_vector_angles(all_pred_xyz, all_true_xyz, pitches, rolls, headings)

    return stats
def main():
    """Main execution function for WAIS angle analysis with enhanced model."""
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join('wais_model_results', timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # Define input files (your existing file lists)
    event_files = ['INSERT_DESIRED_EVENT_FILES']
    header_files = ['INSERT_DESIRED_HEADER_FILES']
    try:
        # Load and process paired data
        print("\nLoading and processing paired WAIS data...")
        voltage_data_h, voltage_data_v, angles, pitches, rolls, headings = load_paired_wais_data(
            event_files, header_files)

        # Save dataset info
        with open(os.path.join(output_dir, 'dataset_info.txt'), 'w') as f:
            f.write("Paired Dataset Statistics:\n")
            f.write("-" * 50 + "\n\n")
            f.write(f"Total paired events: {len(voltage_data_h)}\n")
            f.write(f"WAISH data shape: {voltage_data_h.shape}\n")
            f.write(f"WAIS data shape: {voltage_data_v.shape}\n")
            f.write(f"Angles shape: {angles.shape}\n")

        # Split data
        print("\nSplitting data into training and validation sets...")
        indices = np.arange(len(voltage_data_h))
        train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

        # Create datasets
        train_dataset = PairedWAISDataset(
            voltage_data_h[train_idx], voltage_data_v[train_idx],
            angles[train_idx], pitches[train_idx],
            rolls[train_idx], headings[train_idx]
        )

        val_dataset = PairedWAISDataset(
            voltage_data_h[val_idx], voltage_data_v[val_idx],
            angles[val_idx], pitches[val_idx],
            rolls[val_idx], headings[val_idx]
        )

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)

        # Initialize and train model
        print("\nInitializing and training enhanced model...")
        model = EnhancedWAISCNN()
        model, train_losses, val_losses = train_dual_polarization_model(
            model, train_loader, val_loader, num_epochs=200)

        # Save training history
        plt.figure(figsize=(10, 6))
        plot_training_history(train_losses, val_losses)
        plt.savefig(os.path.join(output_dir, 'training_history.png'))
        plt.close()

        # Save model
        model_path = os.path.join(output_dir, 'final_model.pt')
        torch.save({
            'model_state_dict': model.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses
        }, model_path)

        # Final evaluation
        print("\nPerforming final evaluation...")
        model.eval()
        final_stats = evaluate_model(model, val_loader, device='mps')

        # Save final results
        with open(os.path.join(output_dir, 'final_results.txt'), 'w') as f:
            f.write("Final Model Performance:\n")
            f.write("-" * 50 + "\n\n")
            f.write(f"Mean Azimuth Error: {final_stats['mean_azimuth_error']:.2f}°\n")
            f.write(f"Median Azimuth Error: {final_stats['median_azimuth_error']:.2f}°\n")
            f.write(f"Mean Elevation Error: {final_stats['mean_elevation_error']:.2f}°\n")
            f.write(f"Median Elevation Error: {final_stats['median_elevation_error']:.2f}°\n")

        print("\nAnalysis Complete!")
        print("=" * 50)
        print(f"Results saved to: {output_dir}")

    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
from scipy import constants

if __name__ == "__main__":
    plt.ion()
    main()
    plt.show(block=True)
