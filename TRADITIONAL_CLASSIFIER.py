#%%
import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import seaborn as sns

class WaisThresholdAnalyzer:
    """Analyze WAIS events to determine appropriate classification thresholds."""

    def __init__(self):
        # Antenna mappings (same as before)
        self.antToSurfMap = np.array([
            11, 5, 10, 4, 11, 4, 10, 5, 11, 5, 10, 4, 11, 4, 10, 5,  # Top ring
            9, 3, 8, 2, 8, 3, 9, 2, 9, 3, 8, 2, 8, 3, 9, 2,         # Middle ring
            6, 0, 7, 1, 6, 1, 7, 0, 6, 0, 7, 1, 6, 1, 7, 0          # Bottom ring
        ])

        self.vAntToChan = np.array([
            3, 1, 3, 5, 1, 3, 1, 3, 2, 0, 2, 0, 0, 2, 0, 2,  # Top ring
            1, 3, 1, 3, 3, 1, 3, 1, 0, 2, 0, 2, 2, 0, 2, 0,  # Middle ring
            3, 1, 3, 1, 1, 3, 1, 3, 2, 0, 2, 0, 0, 2, 0, 2   # Bottom ring
        ])

        # Storage for threshold analysis results
        self.wais_voltages = []
        self.wais_neighbor_ratios = []
        self.wais_power_concentrations = []
        self.non_wais_voltages = []
        self.non_wais_neighbor_ratios = []
        self.non_wais_power_concentrations = []

    def process_voltage_data(self, event, timeValues):
        """Process voltage data (same as before)"""
        try:
            data = np.array(event['data'])
            adcOffset = np.array(event['data'].attrs['adcOffset'])
            mvScale = np.array(event['data'].attrs['mvScale'])
            validTimeInds = np.array(event['data'].attrs['validTimeInds'])

            if data.ndim == 3:
                if adcOffset.ndim == 2:
                    adcOffset = adcOffset[..., np.newaxis]
                if mvScale.ndim == 2:
                    mvScale = mvScale[..., np.newaxis]

            data = mvScale * (data - adcOffset)
            v_image = np.zeros((48, 250))

            for ant in range(48):
                surf = self.antToSurfMap[ant]
                chan = self.vAntToChan[ant]

                if surf < data.shape[0] and chan < data.shape[1]:
                    if validTimeInds.ndim > 1:
                        length = min(validTimeInds[surf][1] - validTimeInds[surf][0], 250)
                    else:
                        length = min(validTimeInds[1] - validTimeInds[0], 250)

                    if length > 0:
                        v_image[ant, :length] = data[surf][chan][:length]

            return v_image

        except Exception as e:
            print(f"Error processing voltage data: {str(e)}")
            return None

    def get_neighbors(self, ant_idx):
        """Get physical neighbors (same as before)"""
        ring = ant_idx // 16  # Determine ring (0, 1, or 2)
        ring_start = ring * 16

        # Get left and right neighbors in same ring
        left = ring_start + ((ant_idx - ring_start - 1) % 16)
        right = ring_start + ((ant_idx - ring_start + 1) % 16)
        neighbors = [left, right]

        # Add vertical neighbors
        if ring > 0:  # Has upper neighbor
            neighbors.append(ant_idx - 16)
        if ring < 2:  # Has lower neighbor
            neighbors.append(ant_idx + 16)

        return neighbors

    def analyze_event(self, voltage_data):
        """Extract features from voltage data."""
        # Calculate peak-to-peak voltage for each antenna
        voltage_pp = np.ptp(voltage_data, axis=1)

        # Find maximum signal
        max_antenna = np.argmax(voltage_pp)
        max_voltage = voltage_pp[max_antenna]

        # Get neighbor voltages
        neighbors = self.get_neighbors(max_antenna)
        neighbor_voltages = voltage_pp[neighbors]
        neighbor_ratio = max_voltage / np.mean(neighbor_voltages) if len(neighbor_voltages) > 0 else 0

        # Calculate power concentration
        max_waveform = voltage_data[max_antenna]
        peak_idx = np.argmax(np.abs(max_waveform))
        window = 10  # samples

        start_idx = max(0, peak_idx - window)
        end_idx = min(len(max_waveform), peak_idx + window)

        window_power = np.sum(max_waveform[start_idx:end_idx]**2)
        total_power = np.sum(max_waveform**2)
        power_concentration = window_power / total_power if total_power > 0 else 0

        return {
            'max_voltage': max_voltage,
            'neighbor_ratio': neighbor_ratio,
            'power_concentration': power_concentration
        }

    def analyze_file(self, event_file, header_file):
        """Analyze events and accumulate statistics."""
        print(f"Analyzing {event_file}...")

        with h5py.File(event_file, 'r') as ef, h5py.File(header_file, 'r') as hf:
            run_key = f'run{Path(event_file).stem.replace("event", "")}'

            run_header = hf[run_key]
            run_events = ef[run_key]
            timeValues = np.array(ef['calib']['timeValues'])

            is_wais = run_header['isWAIS'][:]

            for idx in tqdm(range(len(is_wais)), desc="Processing events"):
                event_num = run_header['eventNumber'][idx]
                ev_name = f'ev_{event_num}'

                if ev_name not in run_events:
                    continue

                voltage_data = self.process_voltage_data(run_events[ev_name], timeValues)
                if voltage_data is None:
                    continue

                analysis = self.analyze_event(voltage_data)

                # Store results based on WAIS flag
                if is_wais[idx]:
                    self.wais_voltages.append(analysis['max_voltage'])
                    self.wais_neighbor_ratios.append(analysis['neighbor_ratio'])
                    self.wais_power_concentrations.append(analysis['power_concentration'])
                else:
                    self.non_wais_voltages.append(analysis['max_voltage'])
                    self.non_wais_neighbor_ratios.append(analysis['neighbor_ratio'])
                    self.non_wais_power_concentrations.append(analysis['power_concentration'])

    def determine_thresholds(self, percentile=95):
        """Determine appropriate thresholds based on data distribution."""
        # Convert to arrays for analysis
        wais_v = np.array(self.wais_voltages)
        wais_nr = np.array(self.wais_neighbor_ratios)
        wais_pc = np.array(self.wais_power_concentrations)

        non_wais_v = np.array(self.non_wais_voltages)
        non_wais_nr = np.array(self.non_wais_neighbor_ratios)
        non_wais_pc = np.array(self.non_wais_power_concentrations)

        # Calculate metrics for each feature
        metrics = {}
        for name, wais_data, non_wais_data in [
            ('voltage', wais_v, non_wais_v),
            ('neighbor_ratio', wais_nr, non_wais_nr),
            ('power_concentration', wais_pc, non_wais_pc)
        ]:
            wais_mean = np.mean(wais_data)
            wais_std = np.std(wais_data)
            wais_median = np.median(wais_data)
            wais_percentile = np.percentile(wais_data, 5)  # 5th percentile of WAIS

            non_wais_mean = np.mean(non_wais_data)
            non_wais_std = np.std(non_wais_data)
            non_wais_median = np.median(non_wais_data)
            non_wais_percentile = np.percentile(non_wais_data, 95)  # 95th percentile of non-WAIS

            # Suggestion: use the midpoint between 95th percentile of non-WAIS
            # and 5th percentile of WAIS as threshold
            suggested_threshold = (wais_percentile + non_wais_percentile) / 2

            metrics[name] = {
                'wais_mean': wais_mean,
                'wais_std': wais_std,
                'wais_median': wais_median,
                'wais_5th': wais_percentile,
                'non_wais_mean': non_wais_mean,
                'non_wais_std': non_wais_std,
                'non_wais_median': non_wais_median,
                'non_wais_95th': non_wais_percentile,
                'suggested_threshold': suggested_threshold
            }

        return metrics

    def plot_distributions(self, output_dir=None):
        """Plot distributions of features for WAIS and non-WAIS events."""
        fig, axes = plt.subplots(3, 1, figsize=(12, 15))

        # Plot voltage distributions
        sns.histplot(data=self.non_wais_voltages, ax=axes[0],
                    label='Non-WAIS', alpha=0.5, stat='density')
        sns.histplot(data=self.wais_voltages, ax=axes[0],
                    label='WAIS', alpha=0.5, stat='density')
        axes[0].set_title('Maximum Voltage Distribution')
        axes[0].set_xlabel('Voltage')
        axes[0].legend()

        # Plot neighbor ratio distributions
        sns.histplot(data=self.non_wais_neighbor_ratios, ax=axes[1],
                    label='Non-WAIS', alpha=0.5, stat='density')
        sns.histplot(data=self.wais_neighbor_ratios, ax=axes[1],
                    label='WAIS', alpha=0.5, stat='density')
        axes[1].set_title('Neighbor Ratio Distribution')
        axes[1].set_xlabel('Ratio')
        axes[1].legend()

        # Plot power concentration distributions
        sns.histplot(data=self.non_wais_power_concentrations, ax=axes[2],
                    label='Non-WAIS', alpha=0.5, stat='density')
        sns.histplot(data=self.wais_power_concentrations, ax=axes[2],
                    label='WAIS', alpha=0.5, stat='density')
        axes[2].set_title('Power Concentration Distribution')
        axes[2].set_xlabel('Concentration')
        axes[2].legend()

        plt.tight_layout()

        if output_dir:
            plt.savefig(Path(output_dir) / 'feature_distributions.png')
        plt.show()

def main():
    """Analyze thresholds using real data."""
    # File paths
    event_file = "/Users/georgedoumenis-ramos/PycharmProjects/MSci Research Project/DATA/run130/event130.hdf5"
    header_file = "/Users/georgedoumenis-ramos/PycharmProjects/MSci Research Project/DATA/run130/header130.hdf5"
    output_dir = "threshold_analysis"

    try:
        # Create analyzer
        analyzer = WaisThresholdAnalyzer()

        # Process file
        analyzer.analyze_file(event_file, header_file)

        # Determine thresholds
        metrics = analyzer.determine_thresholds()

        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Save metrics
        with open(Path(output_dir) / 'threshold_metrics.txt', 'w') as f:
            f.write("WAIS Event Threshold Analysis\n")
            f.write("=" * 50 + "\n\n")

            for feature, stats in metrics.items():
                f.write(f"\n{feature.upper()}\n")
                f.write("-" * len(feature) + "\n")
                f.write(f"WAIS events:\n")
                f.write(f"  Mean: {stats['wais_mean']:.2f}\n")
                f.write(f"  Median: {stats['wais_median']:.2f}\n")
                f.write(f"  Std: {stats['wais_std']:.2f}\n")
                f.write(f"  5th percentile: {stats['wais_5th']:.2f}\n")
                f.write(f"\nNon-WAIS events:\n")
                f.write(f"  Mean: {stats['non_wais_mean']:.2f}\n")
                f.write(f"  Median: {stats['non_wais_median']:.2f}\n")
                f.write(f"  Std: {stats['non_wais_std']:.2f}\n")
                f.write(f"  95th percentile: {stats['non_wais_95th']:.2f}\n")
                f.write(f"\nSuggested threshold: {stats['suggested_threshold']:.2f}\n")
                f.write("\n" + "=" * 50 + "\n")

        # Plot distributions
        analyzer.plot_distributions(output_dir)

        print(f"Analysis complete. Results saved to: {output_dir}")

        # Print suggested thresholds
        print("\nSuggested thresholds:")
        for feature, stats in metrics.items():
            print(f"{feature}: {stats['suggested_threshold']:.2f}")

    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
#%%
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import pandas as pd

class WaisClassifier:
    """WAIS event classifier using empirically determined thresholds."""

    def __init__(self, tolerance=0.9):
        """
        Initialize classifier with empirically determined thresholds.
        Args:
            tolerance (float): Factor to adjust thresholds (default 0.9 = 90% of threshold)
        """
        # Antenna mappings from ANITA geometry
        self.antToSurfMap = np.array([
            11, 5, 10, 4, 11, 4, 10, 5, 11, 5, 10, 4, 11, 4, 10, 5,  # Top ring
            9, 3, 8, 2, 8, 3, 9, 2, 9, 3, 8, 2, 8, 3, 9, 2,         # Middle ring
            6, 0, 7, 1, 6, 1, 7, 0, 6, 0, 7, 1, 6, 1, 7, 0          # Bottom ring
        ])

        self.vAntToChan = np.array([
            3, 1, 3, 5, 1, 3, 1, 3, 2, 0, 2, 0, 0, 2, 0, 2,  # Top ring
            1, 3, 1, 3, 3, 1, 3, 1, 0, 2, 0, 2, 2, 0, 2, 0,  # Middle ring
            3, 1, 3, 1, 1, 3, 1, 3, 2, 0, 2, 0, 0, 2, 0, 2   # Bottom ring
        ])

        # Empirically determined criteria
        self.criteria = {
            'min_voltage': 175.71,         # Based on data analysis
            'neighbor_ratio': 1.46,        # Ratio of peak to neighbor voltages
            'power_concentration': 0.31     # Fraction of power in signal window
        }

        self.tolerance = tolerance

    def get_neighbors(self, ant_idx):
        """Get indices of physically adjacent antennas."""
        ring = ant_idx // 16  # Determine ring (0, 1, or 2)
        ring_start = ring * 16

        # Get left and right neighbors in same ring
        left = ring_start + ((ant_idx - ring_start - 1) % 16)
        right = ring_start + ((ant_idx - ring_start + 1) % 16)
        neighbors = [left, right]

        # Add vertical neighbors
        if ring > 0:  # Has upper neighbor
            neighbors.append(ant_idx - 16)
        if ring < 2:  # Has lower neighbor
            neighbors.append(ant_idx + 16)

        return neighbors

    def process_voltage_data(self, event, timeValues):
        """Extract calibrated voltage data from event."""
        try:
            data = np.array(event['data'])
            adcOffset = np.array(event['data'].attrs['adcOffset'])
            mvScale = np.array(event['data'].attrs['mvScale'])
            validTimeInds = np.array(event['data'].attrs['validTimeInds'])

            # Reshape calibration arrays if needed
            if data.ndim == 3:
                if adcOffset.ndim == 2:
                    adcOffset = adcOffset[..., np.newaxis]
                if mvScale.ndim == 2:
                    mvScale = mvScale[..., np.newaxis]

            # Apply calibration
            data = mvScale * (data - adcOffset)

            # Create voltage image
            v_image = np.zeros((48, 250))

            # Fill voltage image using antenna mapping
            for ant in range(48):
                surf = self.antToSurfMap[ant]
                chan = self.vAntToChan[ant]

                if surf < data.shape[0] and chan < data.shape[1]:
                    if validTimeInds.ndim > 1:
                        length = min(validTimeInds[surf][1] - validTimeInds[surf][0], 250)
                    else:
                        length = min(validTimeInds[1] - validTimeInds[0], 250)

                    if length > 0:
                        v_image[ant, :length] = data[surf][chan][:length]

            return v_image

        except Exception as e:
            print(f"Error processing voltage data: {str(e)}")
            return None

    def analyze_event(self, voltage_data):
        """Extract key features from voltage data."""
        # Calculate peak-to-peak voltage for each antenna
        voltage_pp = np.ptp(voltage_data, axis=1)

        # Find maximum signal
        max_antenna = np.argmax(voltage_pp)
        max_voltage = voltage_pp[max_antenna]

        # Calculate neighbor ratio
        neighbors = self.get_neighbors(max_antenna)
        neighbor_voltages = voltage_pp[neighbors]
        neighbor_ratio = max_voltage / np.mean(neighbor_voltages) if len(neighbor_voltages) > 0 else 0

        # Calculate power concentration
        max_waveform = voltage_data[max_antenna]
        peak_idx = np.argmax(np.abs(max_waveform))
        window = 10  # samples

        start_idx = max(0, peak_idx - window)
        end_idx = min(len(max_waveform), peak_idx + window)

        window_power = np.sum(max_waveform[start_idx:end_idx]**2)
        total_power = np.sum(max_waveform**2)
        power_concentration = window_power / total_power if total_power > 0 else 0

        return {
            'max_voltage': max_voltage,
            'max_antenna': max_antenna,
            'neighbor_ratio': neighbor_ratio,
            'power_concentration': power_concentration
        }

    def is_wais_event(self, analysis):
        """Determine if event is WAIS based on empirical thresholds."""
        # Apply criteria with tolerance
        voltage_check = analysis['max_voltage'] > (self.criteria['min_voltage'] * self.tolerance)
        ratio_check = analysis['neighbor_ratio'] > (self.criteria['neighbor_ratio'] * self.tolerance)
        power_check = analysis['power_concentration'] > (self.criteria['power_concentration'] * self.tolerance)

        # Return True only if all criteria are met
        return voltage_check and ratio_check and power_check

    def classify_events(self, event_file, header_file, output_dir=None):
        """Classify all events in a run."""
        results = []
        print(f"Processing {event_file}...")

        with h5py.File(event_file, 'r') as ef, h5py.File(header_file, 'r') as hf:
            run_key = f'run{Path(event_file).stem.replace("event", "")}'

            run_header = hf[run_key]
            run_events = ef[run_key]
            timeValues = np.array(ef['calib']['timeValues'])

            is_wais = run_header['isWAIS'][:]

            for idx in tqdm(range(len(is_wais)), desc="Classifying events"):
                event_num = run_header['eventNumber'][idx]
                ev_name = f'ev_{event_num}'

                if ev_name not in run_events:
                    continue

                # Process voltage data
                voltage_data = self.process_voltage_data(run_events[ev_name], timeValues)
                if voltage_data is None:
                    continue

                # Analyze and classify
                analysis = self.analyze_event(voltage_data)
                predicted_wais = self.is_wais_event(analysis)

                # Store result
                result = {
                    'event_number': event_num,
                    'predicted_wais': predicted_wais,
                    'true_wais': bool(is_wais[idx]),
                    **analysis
                }

                results.append(result)

        if output_dir:
            self.save_results(results, output_dir)

        return results

    def save_results(self, results, output_dir):
        """Save classification results and plots."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Extract predictions and true labels
        predictions = [r['predicted_wais'] for r in results]
        true_labels = [r['true_wais'] for r in results]

        # Calculate confusion matrix
        cm = confusion_matrix(true_labels, predictions)

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(Path(output_dir) / 'confusion_matrix.png')
        plt.close()

        # Save classification report
        with open(Path(output_dir) / 'classification_report.txt', 'w') as f:
            f.write("WAIS Classification Results\n")
            f.write("=" * 50 + "\n\n")
            f.write("Thresholds Used:\n")
            for key, value in self.criteria.items():
                f.write(f"{key}: {value:.2f}\n")
            f.write(f"Tolerance factor: {self.tolerance}\n\n")

            f.write(classification_report(true_labels, predictions))

            # Additional metrics
            tn, fp, fn, tp = cm.ravel()
            total = tn + fp + fn + tp

            f.write("\nDetailed Metrics:\n")
            f.write(f"Total events: {total}\n")
            f.write(f"True WAIS events: {tp + fn}\n")
            f.write(f"Predicted WAIS events: {tp + fp}\n")

            accuracy = (tp + tn) / total
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            f.write(f"\nAccuracy: {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n")

            # Plot feature distributions
            self.plot_feature_distributions(results, output_dir)

    def plot_feature_distributions(self, results, output_dir):
        """Plot distributions of features by class."""
        # Convert results to DataFrame
        df = pd.DataFrame(results)

        fig, axes = plt.subplots(3, 1, figsize=(12, 15))

        # Voltage distribution
        sns.histplot(data=df, x='max_voltage', hue='true_wais',
                    multiple="layer", bins=50, ax=axes[0])
        axes[0].axvline(self.criteria['min_voltage'], color='r', linestyle='--')
        axes[0].set_title('Maximum Voltage Distribution')

        # Neighbor ratio distribution
        sns.histplot(data=df, x='neighbor_ratio', hue='true_wais',
                    multiple="layer", bins=50, ax=axes[1])
        axes[1].axvline(self.criteria['neighbor_ratio'], color='r', linestyle='--')
        axes[1].set_title('Neighbor Ratio Distribution')

        # Power concentration distribution
        sns.histplot(data=df, x='power_concentration', hue='true_wais',
                    multiple="layer", bins=50, ax=axes[2])
        axes[2].axvline(self.criteria['power_concentration'], color='r', linestyle='--')
        axes[2].set_title('Power Concentration Distribution')

        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'feature_distributions.png')
        plt.close()

def main():
    """Run WAIS classification on a dataset."""
    # File paths
    event_file = []
    header_file = []
    output_dir = "wais_classification_results"

    try:
        # Create classifier with default tolerance
        classifier = WaisClassifier(tolerance=0.9)

        # Classify events
        results = classifier.classify_events(event_file, header_file, output_dir)

        print(f"\nClassification complete. Results saved to: {output_dir}")

    except Exception as e:
        print(f"Error during classification: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
