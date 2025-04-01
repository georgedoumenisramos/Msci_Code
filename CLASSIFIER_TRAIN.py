import numpy as np
import h5py
import random
from pathlib import Path


def load_wais_classification_data(event_files, header_files, n_non_wais_samples=None):
    """
    Load WAIS and non-WAIS events for binary classification.

    Parameters:
        event_files (list): List of paths to event HDF5 files
        header_files (list): List of paths to header HDF5 files
        n_non_wais_samples (int, optional): Number of non-WAIS events to sample

    Returns:
        tuple: (voltage_data, labels)
            - voltage_data: numpy array of shape (n_events, 48, 250)
            - labels: binary array (1 for WAIS, 0 for non-WAIS)
    """
    # First pass: Count WAIS events and collect non-WAIS indices
    total_wais = 0
    non_wais_locations = []  # List of (file_idx, event_idx) tuples

    print("First pass: Counting WAIS events and locating non-WAIS events...")
    for file_idx, (event_file, header_file) in enumerate(zip(event_files, header_files)):
        run_number = Path(event_file).stem.replace('event', '')
        run_key = f'run{run_number}'

        with h5py.File(header_file, 'r') as hf:
            run_header = hf[run_key]
            is_wais = run_header['isWAIS'][:]

            # Count WAIS events
            total_wais += np.sum(is_wais)

            # Store non-WAIS locations
            non_wais_indices = np.where(~is_wais)[0]
            non_wais_locations.extend([(file_idx, idx) for idx in non_wais_indices])

    print(f"Found {total_wais} WAIS events")
    print(f"Found {len(non_wais_locations)} non-WAIS events")

    # Determine number of non-WAIS events to sample
    if n_non_wais_samples is None:
        n_non_wais_samples = total_wais
    n_non_wais_samples = min(n_non_wais_samples, len(non_wais_locations))

    # Sample non-WAIS locations
    sampled_non_wais = random.sample(non_wais_locations, n_non_wais_samples)

    # Group sampled events by file for efficient loading
    events_by_file = {}
    for file_idx, event_idx in sampled_non_wais:
        if file_idx not in events_by_file:
            events_by_file[file_idx] = []
        events_by_file[file_idx].append(event_idx)

    # Process events
    wais_data = []
    non_wais_data = []

    print("\nProcessing events...")
    for file_idx, (event_file, header_file) in enumerate(zip(event_files, header_files)):
        run_number = Path(event_file).stem.replace('event', '')
        run_key = f'run{run_number}'

        with h5py.File(event_file, 'r') as ef, h5py.File(header_file, 'r') as hf:
            run_header = hf[run_key]
            run_events = ef[run_key]
            timeValues = np.array(ef['calib']['timeValues'])

            is_wais = run_header['isWAIS'][:]

            # Process all WAIS events
            wais_indices = np.where(is_wais)[0]
            print(f"\nProcessing {len(wais_indices)} WAIS events from {run_key}")
            for idx in tqdm(wais_indices, desc="WAIS events"):
                try:
                    voltage_data = process_single_event(
                        idx, run_header, run_events, timeValues
                    )
                    if voltage_data is not None:
                        wais_data.append(voltage_data)
                except Exception as e:
                    print(f"Error processing WAIS event {idx}: {str(e)}")
                    continue

            # Process only sampled non-WAIS events for this file
            if file_idx in events_by_file:
                non_wais_indices = events_by_file[file_idx]
                print(f"Processing {len(non_wais_indices)} sampled non-WAIS events from {run_key}")
                for idx in tqdm(non_wais_indices, desc="non-WAIS events"):
                    try:
                        voltage_data = process_single_event(
                            idx, run_header, run_events, timeValues
                        )
                        if voltage_data is not None:
                            non_wais_data.append(voltage_data)
                    except Exception as e:
                        print(f"Error processing non-WAIS event {idx}: {str(e)}")
                        continue

    # Convert to numpy arrays
    wais_data = np.array(wais_data)
    non_wais_data = np.array(non_wais_data)

    # Combine data and create labels
    voltage_data = np.concatenate([wais_data, non_wais_data])
    labels = np.concatenate([
        np.ones(len(wais_data)),
        np.zeros(len(non_wais_data))
    ])

    print("\nFinal dataset statistics:")
    print(f"Total events: {len(voltage_data)}")
    print(f"WAIS events: {len(wais_data)}")
    print(f"Non-WAIS events: {len(non_wais_data)}")
    print(f"Voltage data shape: {voltage_data.shape}")

    return voltage_data, labels


def getTimesAndMillivolts(event, timeValues):
    """Get calibrated voltages from event data."""
    adcOffset = np.array(event['data'].attrs['adcOffset'])
    mvScale = np.array(event['data'].attrs['mvScale'])
    timeOffset = np.array(event['data'].attrs['timeOffset'])
    timeScale = np.array(event['data'].attrs['timeScale'])
    validTimeInds = np.array(event['data'].attrs['validTimeInds'])
    chips = np.array(event['data'].attrs['chips'])
    data = np.array(event['data'])

    adcOffset = np.reshape(adcOffset, (adcOffset.shape[0], adcOffset.shape[1], -1))
    mvScale = np.reshape(mvScale, (mvScale.shape[0], mvScale.shape[1], -1))
    timeOffset = np.reshape(timeOffset, (timeOffset.shape[0], timeOffset.shape[1], -1))
    timeScale = np.reshape(timeScale, (timeScale.shape[0], timeScale.shape[1], -1))

    data = mvScale * (data - adcOffset)
    times = np.zeros((12, 9, 250))
    N = np.zeros((12, 9), dtype=int)

    for surf in range(12):
        for chan in range(9):
            valid_length = validTimeInds[surf][1] - validTimeInds[surf][0]
            times[surf][chan][:valid_length] = timeValues[surf][chips[surf]][
                                               validTimeInds[surf][0]:validTimeInds[surf][1]
                                               ]
            times[surf][chan] = times[surf][chan] - times[surf][chan][0]
            N[surf][chan] = valid_length

    return N, times, data


def process_single_event(idx, run_header, run_events, timeValues):
    """Process a single event and return voltage data."""
    event_num = run_header['eventNumber'][idx]
    ev_name = f'ev_{event_num}'

    if ev_name not in run_events:
        return None

    event = run_events[ev_name]

    # Get voltage data
    N, times, voltages = getTimesAndMillivolts(event, timeValues)

    # Process into standard format
    v_image = np.zeros((48, 250))

    # Antenna mappings
    ant_to_surf_map = np.array([i // 4 for i in range(48)])
    v_ant_to_chan = np.array([i % 4 for i in range(48)])

    # Fill voltage image
    for ant in range(48):
        surf = ant_to_surf_map[ant]
        chan = v_ant_to_chan[ant]
        if surf < voltages.shape[0] and chan < voltages.shape[1]:
            length = min(N[surf][chan], 250)
            v_image[ant, :length] = voltages[surf][chan][:length]

    return v_image


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from tqdm import tqdm


class WAISDataset(Dataset):
    """Dataset for WAIS event classification"""

    def __init__(self, voltage_data, labels):
        self.voltage_data = torch.FloatTensor(voltage_data)
        self.labels = torch.FloatTensor(labels)

        # Normalize each event's voltage data
        for i in range(len(self.voltage_data)):
            vdata = self.voltage_data[i]
            vmin, vmax = vdata.min(), vdata.max()
            if vmax > vmin:
                self.voltage_data[i] = 2 * (vdata - vmin) / (vmax - vmin) - 1

        # Add channel dimension for CNN
        self.voltage_data = self.voltage_data.unsqueeze(1)

    def __len__(self):
        return len(self.voltage_data)

    def __getitem__(self, idx):
        return {
            'voltage': self.voltage_data[idx],
            'label': self.labels[idx]
        }


class WAISClassifier(nn.Module):
    """CNN for WAIS event classification"""

    def __init__(self, dropout_rate=0.5):
        super().__init__()

        # CNN layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2)
        )

        # Calculate size after convolutions
        with torch.no_grad():
            x = torch.randn(1, 1, 48, 250)
            x = self.conv_layers(x)
            conv_size = x.numel()

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_size, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc_layers(x).squeeze()


def train_wais_classifier(voltage_data, labels, batch_size=32, num_epochs=10,
                          val_split=0.2, output_dir='wais_outputs'):
    """Train WAIS classifier"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create dataset
    dataset = WAISDataset(voltage_data, labels)

    # Split dataset
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model
    model = WAISClassifier().to(device)

    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)

    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            voltage = batch['voltage'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(voltage)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

        train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                voltage = batch['voltage'].to(device)
                labels = batch['label'].to(device)

                outputs = model(voltage)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

                # Store predictions and labels for metrics
                all_predictions.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total

        # Update learning rate
        scheduler.step(val_loss)

        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        # Print progress
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc
            }, os.path.join(output_dir, 'best_model.pt'))

            # Calculate and save additional metrics for best model
            all_predictions = np.array(all_predictions)
            all_labels = np.array(all_labels)

            # ROC curve
            fpr, tpr, _ = roc_curve(all_labels, all_predictions)
            roc_auc = auc(fpr, tpr)

            # Plot ROC curve
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2,
                     label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - Epoch {epoch + 1}')
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(output_dir, f'roc_curve_epoch_{epoch + 1}.png'))
            plt.close()

            # Confusion matrix
            pred_labels = (all_predictions > 0.5).astype(int)
            cm = confusion_matrix(all_labels, pred_labels)

            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - Epoch {epoch + 1}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(os.path.join(output_dir, f'confusion_matrix_epoch_{epoch + 1}.png'))
            plt.close()

            # Save metrics to file
            metrics_file = os.path.join(output_dir, 'best_model_metrics.txt')
            with open(metrics_file, 'w') as f:
                f.write(f"Best Model Metrics (Epoch {epoch + 1})\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Validation Loss: {val_loss:.4f}\n")
                f.write(f"Validation Accuracy: {val_acc:.2f}%\n")
                f.write(f"ROC AUC: {roc_auc:.4f}\n\n")

                f.write("Confusion Matrix:\n")
                f.write(f"TN: {cm[0, 0]}, FP: {cm[0, 1]}\n")
                f.write(f"FN: {cm[1, 0]}, TP: {cm[1, 1]}\n\n")

                precision = cm[1, 1] / (cm[1, 1] + cm[0, 1])
                recall = cm[1, 1] / (cm[1, 1] + cm[1, 0])
                f1 = 2 * (precision * recall) / (precision + recall)

                f.write("Additional Metrics:\n")
                f.write(f"Precision: {precision:.4f}\n")
                f.write(f"Recall: {recall:.4f}\n")
                f.write(f"F1 Score: {f1:.4f}\n")

        # Plot training progress
        if (epoch + 1) % 5 == 0:
            plt.figure(figsize=(12, 4))

            # Loss plot
            plt.subplot(1, 2, 1)
            plt.plot(history['train_loss'], label='Train')
            plt.plot(history['val_loss'], label='Validation')
            plt.title('Loss vs Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()

            # Accuracy plot
            plt.subplot(1, 2, 2)
            plt.plot(history['train_acc'], label='Train')
            plt.plot(history['val_acc'], label='Validation')
            plt.title('Accuracy vs Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            plt.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'training_progress_epoch_{epoch + 1}.png'))
            plt.close()

    # Save final training history
    np.save(os.path.join(output_dir, 'training_history.npy'), history)

    return model, history


import os
from datetime import datetime
import numpy as np
import torch


def main():
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # Define file paths
    event_files = []

    header_files = []

    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join('outputs', f'wais_training_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Load dataset
        print("\nLoading and processing data...")
        voltage_data, labels = load_wais_classification_data(
            event_files=event_files,
            header_files=header_files,
            n_non_wais_samples=2000  # Or adjust as needed
        )

        # Save dataset statistics
        with open(os.path.join(output_dir, 'dataset_info.txt'), 'w') as f:
            f.write("Dataset Statistics:\n")
            f.write("-" * 50 + "\n\n")
            f.write(f"Total events: {len(voltage_data)}\n")
            f.write(f"WAIS events: {int(np.sum(labels))}\n")
            f.write(f"Non-WAIS events: {int(len(labels) - np.sum(labels))}\n")
            f.write(f"Voltage data shape: {voltage_data.shape}\n")

            # Add file information
            f.write("\nInput Files:\n")
            for ef, hf in zip(event_files, header_files):
                f.write(f"Event: {ef}\n")
                f.write(f"Header: {hf}\n")

        # Training parameters
        training_params = {
            'batch_size': 16,
            'num_epochs': 20,
            'val_split': 0.2,
            'output_dir': output_dir
        }

        # Save training parameters
        with open(os.path.join(output_dir, 'training_params.txt'), 'w') as f:
            for key, value in training_params.items():
                f.write(f"{key}: {value}\n")

        # Train model
        print("\nStarting model training...")
        model, history = train_wais_classifier(
            voltage_data=voltage_data,
            labels=labels,
            **training_params
        )

        print("\nTraining complete!")
        print(f"Results saved to: {output_dir}")

    except Exception as e:
        print(f"\nError during execution: {str(e)}")
        import traceback
        traceback.print_exc()
# if __name__ == "__main__":
#     main()
