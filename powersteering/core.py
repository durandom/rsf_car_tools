import os
import json
import re
import shutil
import numpy as np
from datetime import datetime
from io import StringIO
from configobj import ConfigObj
from typing import List, Tuple, Dict, Optional, TYPE_CHECKING
from collections import defaultdict
from loguru import logger
import plotext as plt
from .models import Car
from .renderer import ConsoleRenderer

from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans



class PowerSteering:
    def __init__(self, rsf_path: str, global_ffb: bool = False):
        """Initialize RSF configuration handler

        Args:
            rsf_path (str): Path to RSF installation directory
            global_ffb (bool): Consider global FFB values when detecting custom FFB
        """
        self.rsf_path = rsf_path
        self.global_ffb = global_ffb
        self.features = ['weight', 'steering_wheel', 'drive_train']  # Default features

        # Define required files
        self.personal_ini = os.path.join(rsf_path, 'rallysimfans_personal.ini')
        self.rbr_ini = os.path.join(rsf_path, 'RichardBurnsRally.ini')
        self.cars_json = os.path.join(rsf_path, 'rsfdata', 'cache', 'cars.json')
        self.cars_data_json = os.path.join(rsf_path, 'rsfdata', 'cache', 'cars_data.json')

        self._validate_files()
        self.cars: Dict[str, Car] = {}
        self.undriven_cars: Dict[str, Car] = {}  # Track cars found in cars.json but not in personal.ini
        # Global FFB settings from RBR
        self.ffb_tarmac = 0
        self.ffb_gravel = 0
        self.ffb_snow = 0
        self._load_rbr_ini()
        self._load_personal_ini()
        self._load_cars_json()
        self._load_cars_data_json()
        self.drive_map = self._build_drive_map()

    def _validate_files(self) -> None:
        """Validate that all required RSF configuration files exist.

        Checks for the presence of:
        - rallysimfans_personal.ini
        - RichardBurnsRally.ini
        - cars.json
        - cars_data.json

        Raises:
            FileNotFoundError: If any required files are missing
        """
        missing_files = []

        for filepath in [self.personal_ini, self.rbr_ini, self.cars_json, self.cars_data_json]:
            if not os.path.exists(filepath):
                missing_files.append(filepath)

        if missing_files:
            raise FileNotFoundError(
                "Missing required files:\n" +
                "\n".join(f"- {f}" for f in missing_files)
            )

    def config_parser(self, file: str) -> ConfigObj:
        """Parse an INI configuration file with special handling for RBR format.

        Handles BOM markers and converts semicolon comments to hash style.
        Maintains compatibility with rbr_pacenote_plugin parsing style.

        Args:
            file: Path to the INI file to parse

        Returns:
            ConfigObj: Parsed configuration object

        Raises:
            Exception: If file cannot be parsed
        """
        try:
            with open(file, 'r', encoding='utf-8') as f:
                # strip bom and convert ; to # for comments
                contents = f.read()
                contents = contents.replace('\ufeff', '')
                # Corrects ";" at the start of any line to "#" throughout the file.
                # file_contents = re.sub(r'^\s*;', '#', file_contents, flags=re.MULTILINE)
                contents = re.sub(r';', '#', contents, flags=re.MULTILINE)

            # Convert contents to list of strings and pass to ConfigObj
            config = ConfigObj(StringIO(contents), encoding='utf-8', file_error=True)
            return config

        except Exception as e:
            logger.critical(f"Error parsing {file}: {str(e)}")
            exit(1)

    def _load_cars_data_json(self) -> None:
        """Load cars_data.json and add technical data to existing Car objects"""
        encodings = ['utf-8', 'latin1', 'cp1252']

        for encoding in encodings:
            try:
                with open(self.cars_data_json, 'r', encoding=encoding) as f:
                    cars_data_json = json.load(f)
                    for car_data in cars_data_json:
                        car_id = car_data['car_id']
                        if car_id in self.cars:
                            car = self.cars[car_id]
                            # Add technical data attributes to existing Car object
                        elif car_id in self.undriven_cars:
                            car = self.undriven_cars[car_id]
                        else:
                            logger.error(f"Car {car_id} found in cars_data.json but not in personal.ini")
                            continue
                        # Convert numeric values at load time
                        try:
                            power = car_data.get('power', '0')
                            car.power = int(power) if power else 0
                        except ValueError:
                            car.power = 0

                        try:
                            torque = car_data.get('torque', '0')
                            car.torque = int(torque) if torque else 0
                        except ValueError:
                            car.torque = 0

                        car.drive_train = car_data.get('drive_train', '')
                        car.engine = car_data.get('engine', '')
                        car.transmission = car_data.get('transmission', '')

                        try:
                            weight = car_data.get('weight', '0')
                            car.weight = int(weight.lower().replace('kg', '').strip()) if weight else 0
                        except ValueError:
                            car.weight = 0

                        car.wdf = car_data.get('wdf', '')

                        try:
                            steering = car_data.get('steering_wheel', '0')
                            car.steering_wheel = int(steering.replace('°', '').strip()) if steering else 0
                        except ValueError:
                            car.steering_wheel = 0
                        car.skin = car_data.get('skin', '')
                        car.model = car_data.get('model', '')
                        car.year = car_data.get('year', '')
                        car.shifter_type = car_data.get('shifterType', '')
                        logger.debug(f"Added technical data to car {car_id}")
                    logger.debug(f"Successfully loaded cars_data.json with {encoding} encoding")
                    return
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.error(f"Error loading cars_data.json: {str(e)}")
                return

        logger.error("Failed to load cars_data.json with any supported encoding")

    def _load_rbr_ini(self) -> None:
        """Load global FFB settings from RichardBurnsRally.ini"""
        config = self.config_parser(self.rbr_ini)

        if 'NGP' in config:
            ngp_section = config['NGP']
            # make sure ngp_section is a dictionary
            if not isinstance(ngp_section, dict):
                return

            self.ffb_tarmac = int(ngp_section.get('ForceFeedbackSensitivityTarmac', 0))
            self.ffb_gravel = int(ngp_section.get('ForceFeedbackSensitivityGravel', 0))
            self.ffb_snow = int(ngp_section.get('ForceFeedbackSensitivitySnow', 0))
            logger.debug(f"Loaded global FFB settings - Tarmac: {self.ffb_tarmac}, Gravel: {self.ffb_gravel}, Snow: {self.ffb_snow}")

    def _load_personal_ini(self) -> None:
        """Load car configurations from personal.ini"""
        config = self.config_parser(self.personal_ini)

        for section_name in config:
            if not section_name.startswith('car'):
                continue

            car_id = section_name[3:]  # Remove 'car' prefix
            # Get section data and ensure string types
            section = config[section_name]
            # make sure section is a dictionary
            if not isinstance(section, dict):
                continue
            car_data = {str(k): str(v) for k, v in section.items()}
            logger.debug(f"Loaded car configuration: {car_id} - {car_data}")
            if self.global_ffb:
                self.cars[car_id] = Car(car_id, car_data,
                                      self.ffb_tarmac, self.ffb_gravel, self.ffb_snow)
            else:
                self.cars[car_id] = Car(car_id, car_data)

    def _load_cars_json(self) -> None:
        """Load cars.json and add data to existing Car objects"""
        try:
            with open(self.cars_json, 'r', encoding='utf-8') as f:
                cars_json = json.load(f)
                for car_json in cars_json:
                    car_id = car_json['id']
                    if car_id in self.cars:
                        car = self.cars[car_id]
                        logger.debug(f"Added JSON data to car {car_id}")
                    else:
                        logger.debug(f"Car {car_id} found in cars.json but not in personal.ini")
                        # Create a Car object for undriven cars too
                        car = Car(car_id, {})  # Empty config since not in personal.ini
                        self.undriven_cars[car_id] = car

                    if not car.name:
                        car.name = car_json.get('name', '')
                    car.path = car_json.get('path', '')
                    car.hash = car_json.get('hash', '')
                    car.carmodel_id = car_json.get('carmodel_id', '')
                    car.user_id = car_json.get('user_id', '')
                    car.base_group_id = car_json.get('base_group_id', '')
                    car.ngp = car_json.get('ngp', '')
                    car.custom_setups = car_json.get('custom_setups', '')
                    car.rev = car_json.get('rev', '')
                    car.audio = car_json.get('audio')
                    car.audio_hash = car_json.get('audio_hash', '')
        except Exception as e:
            logger.error(f"Error loading cars.json: {str(e)}")


    def format_car_details(self, car: Car) -> str:
        """Format car details for display

        Args:
            car_data: Dictionary containing car data

        Returns:
            Formatted string with key car details
        """
        car_id = car.id
        model = car.model
        year = car.year
        drive_train = car.drive_train

        return f"[ID: {car_id}] {model} {year} - {drive_train}"


    def _extract_feature_values(self, car: Car) -> Optional[List[float]]:
        """Extract feature values from a car.

        Args:
            car: Car object to extract features from

        Returns:
            List of feature values if successful, None if any features are invalid
        """
        try:
            feature_values = []
            for feature in self.features:
                if feature == 'drive_train':
                    value = self.drive_map.get(getattr(car, feature).upper(), 0)
                else:
                    value = getattr(car, feature)
                if not value or value <= 0:
                    raise ValueError(f"Invalid {feature} value")
                feature_values.append(value)
            return feature_values
        except (ValueError, AttributeError) as e:
            logger.warning(f"Could not extract features from car {car.id}: {str(e)}")
            return None

    def prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare feature and target arrays for FFB prediction model training.

        Extracts relevant features (weight, steering angle, drivetrain) and
        FFB settings (tarmac, gravel, snow) from cars with custom configurations.
        Filters out cars with invalid or missing data.

        Returns:
            tuple: (features, targets) where:
                features: ndarray of shape (n_samples, n_features) containing
                         weight, steering angle, and drivetrain values
                targets: ndarray of shape (n_samples, 3) containing FFB values
                        for tarmac, gravel and snow surfaces
        """
        features = []
        targets = []

        for car in self.cars.values():
            if car.has_custom_ffb():
                feature_values = self._extract_feature_values(car)
                if feature_values:
                    features.append(feature_values)
                    targets.append([car.ffb_tarmac, car.ffb_gravel, car.ffb_snow])

        return np.array(features), np.array(targets)

    def create_ffb_model(self) -> Pipeline:
        """Create a regression pipeline for FFB prediction.

        Builds a scikit-learn pipeline with:
        1. StandardScaler for feature normalization
        2. LinearRegression for prediction

        Returns:
            Pipeline: Configured scikit-learn pipeline ready for training
        """
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression(fit_intercept=True))
        ])

        return model

    def train_ffb_models(self):
        """Train separate models for each surface type"""
        X, y = self.prepare_training_data()
        if len(X) == 0:
            logger.error("No valid training data found")
            return None

        # Scale target values
        y_scaler = StandardScaler()
        y_scaled = y_scaler.fit_transform(y)

        # Use consistent random state for reproducibility
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_scaled, test_size=0.2, random_state=42
        )

        # Train separate model for each surface
        models = {}
        surface_names = ['tarmac', 'gravel', 'snow']

        for i, surface in enumerate(surface_names):
            model = self.create_ffb_model()
            model.fit(X_train, y_train[:, i])
            score = model.score(X_test, y_test[:, i])
            logger.info(f"{surface.title()} model R² score: {score:.3f}")
            models[surface] = model

        # Store the y_scaler for predictions
        self.y_scaler = y_scaler
        return models

    def _build_drive_map(self) -> Dict[str, int]:
        """Build mapping of drivetrains to numeric values based on available cars"""
        drive_types = sorted(set(car.drive_train.upper() for car in self.cars.values()
                               if car.drive_train))
        drive_map = {dt: idx + 1 for idx, dt in enumerate(drive_types)}
        logger.debug(f"Built drivetrain mapping: {drive_map}")
        return drive_map

    def predict_ffb_settings(self, car: Car, models: dict) -> Tuple[int, int, int]:
        """Predict FFB settings for a car using trained models"""
        try:
            # Extract features
            feature_values = []
            for feature in self.features:
                if feature == 'drive_train':
                    value = self.drive_map.get(getattr(car, feature).upper(), 0)
                else:
                    value = getattr(car, feature)
                if not value or value <= 0:
                    return (self.ffb_tarmac, self.ffb_gravel, self.ffb_snow)
                feature_values.append(value)

            features = np.array([feature_values])

            # Get scaled predictions for each surface
            scaled_predictions = np.zeros((1, 3))
            for i, surface in enumerate(['tarmac', 'gravel', 'snow']):
                scaled_predictions[0, i] = models[surface].predict(features)[0]

            # Inverse transform to get original scale
            predictions = self.y_scaler.inverse_transform(scaled_predictions)[0]

            # Clip and round predictions
            predictions = [max(50, min(2000, int(round(p)))) for p in predictions]
            ffb_tarmac, ffb_gravel, ffb_snow = predictions

            return (ffb_tarmac, ffb_gravel, ffb_snow)

        except (ValueError, AttributeError) as e:
            logger.warning(f"Could not predict FFB settings for car {car.id}: {str(e)}")
            return (self.ffb_tarmac, self.ffb_gravel, self.ffb_snow)

    def _find_optimal_clusters(self, features_scaled: np.ndarray) -> int:
        """Find optimal number of clusters using elbow method.

        Args:
            features_scaled: Normalized feature matrix

        Returns:
            Optimal number of clusters
        """
        max_clusters = min(10, len(features_scaled) // 5)  # Don't try more than n/5 clusters
        distortions = []

        for k in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(features_scaled)
            distortions.append(kmeans.inertia_)

        # Find elbow point using second derivative
        diffs = np.diff(distortions, 2)  # Second derivative
        elbow_idx = int(np.argmin(diffs)) + 2  # Add 2 due to double diff

        return elbow_idx + 1  # Add 1 since we started at k=1

    def select_training_sample(self, min_cars_per_cluster: int = 3) -> List[Car]:
        """Select a representative sample of cars using clustering.

        Args:
            min_cars_per_cluster: Minimum number of cars to select from each cluster

        Returns:
            List of selected Car objects
        """
        # Prepare feature matrix for clustering
        features = []
        valid_cars = []

        for car in self.cars.values():
            feature_values = self._extract_feature_values(car)
            if feature_values:
                features.append(feature_values)
                valid_cars.append(car)

        if not features:
            logger.error("No valid cars found for clustering")
            return []

        # Normalize features to 0-1 range
        features = np.array(features)
        scaler = MinMaxScaler()
        features_scaled = scaler.fit_transform(features)

        # Find optimal number of clusters
        n_clusters = self._find_optimal_clusters(features_scaled)
        logger.debug(f"Determined optimal number of clusters: {n_clusters}")

        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(features_scaled)

        # Select cars from each cluster
        selected_cars = []
        for cluster_id in range(n_clusters):
            # Get indices of cars in this cluster
            cluster_indices = np.where(clusters == cluster_id)[0]
            cluster_size = len(cluster_indices)

            # Select specified number of cars from cluster
            n_select = min(min_cars_per_cluster, cluster_size)  # Can't select more cars than exist in cluster

            # Get cluster center
            cluster_center = kmeans.cluster_centers_[cluster_id]

            # Calculate distances to cluster center for all points in this cluster
            cluster_points = features_scaled[cluster_indices]
            distances = np.linalg.norm(cluster_points - cluster_center, axis=1)

            # Get indices of n_select closest points to center
            closest_in_cluster = np.argsort(distances)[:n_select]
            selected_indices = cluster_indices[closest_in_cluster]

            for idx in selected_indices:
                car = valid_cars[idx]
                car.cluster = cluster_id  # Add cluster ID to car object
                selected_cars.append(car)

        logger.info(f"Selected {len(selected_cars)} cars from {n_clusters} clusters")
        return selected_cars

    def get_cluster_data(self, selected_cars: List[Car]) -> Dict:
        """Get cluster data for selected cars

        Args:
            selected_cars: List of selected Car objects

        Returns:
            Dict mapping cluster IDs to lists of cars
        """
        cluster_data = defaultdict(list)
        features = []
        valid_cars = []

        for car in self.cars.values():
            feature_values = self._extract_feature_values(car)
            if feature_values:
                features.append(feature_values)
                valid_cars.append(car)

        features = np.array(features)
        scaler = MinMaxScaler()
        features_scaled = scaler.fit_transform(features)
        n_clusters = self._find_optimal_clusters(features_scaled)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(features_scaled)

        for car, cluster_id in zip(valid_cars, clusters):
            car.cluster = cluster_id
            cluster_data[cluster_id].append(car)

        return cluster_data

    def set_features(self, features: List[str]) -> None:
        """Set the features to use for training and prediction

        Args:
            features: List of feature names (must be attributes of Car class)
        """
        self.features = features
        logger.info(f"Set features to: {features}")

    def predict_all_ffb_settings(self, models: dict) -> List[Tuple[Car, Tuple[int, int, int]]]:
        """Generate FFB predictions for all cars

        Args:
            models: Dictionary of trained models for each surface

        Returns:
            List of tuples containing (Car, (tarmac_ffb, gravel_ffb, snow_ffb))
        """
        if not models:
            logger.error("No trained models available")
            return []

        cars_with_predictions = []
        for car_id, car in self.cars.items():
            predictions = self.predict_ffb_settings(car, models)
            cars_with_predictions.append((car, predictions))

        return cars_with_predictions

    def write_ai_ffb_file(self, cars_with_predictions: List[Tuple[Car, Tuple[int, int, int]]], output_file: str) -> None:
        """Write predicted FFB settings to a new personal.ini file

        Args:
            cars_with_predictions: List of (Car, predictions) tuples from predict_all_ffb_settings
            output_file: Path to output file
        """
        logger.info(f"Starting FFB file generation: reading from {self.personal_ini}")
        logger.info(f"Writing predictions to {output_file}")

        cars_processed = 0
        cars_modified = 0

        try:
            with open(self.personal_ini, 'r', encoding='utf-8') as infile, \
                 open(output_file, 'w', encoding='utf-8') as outfile:

                current_car_id = None
                current_car = None
                predictions = None
                for line in infile:
                    # Remove BOM if present
                    line = line.replace('\ufeff', '')

                    # Check for any section header
                    if line.strip().startswith('['):
                        # Reset car tracking for non-car sections
                        if not line.strip().startswith('[car'):
                            current_car_id = None
                            current_car = None
                            predictions = None
                        else:
                            # Handle car section
                            current_car_id = line.strip()[4:-1]  # Extract ID between [car and ]
                            current_car = self.cars.get(current_car_id)
                            cars_processed += 1

                            if current_car and not current_car.has_custom_ffb():
                                # Find predictions for this car from cars_with_predictions
                                predictions = next((pred for car, pred in cars_with_predictions
                                                 if car.id == current_car_id), None)
                                if predictions:
                                    cars_modified += 1
                                    logger.debug(f"Set predicted FFB for car {current_car_id}: "
                                               f"T:{predictions[0]} G:{predictions[1]} S:{predictions[2]}")
                                else:
                                    logger.critical(f"Car {current_car_id} found in personal.ini but not in predictions")
                                    raise Exception(f"Car {current_car_id} found in personal.ini but not in predictions")
                            else:
                                logger.warning(f"Car {current_car_id} found in personal.ini but not in cars.json")
                                predictions = None

                    # Check for FFB settings if we have predictions for current car
                    elif current_car:
                        if current_car.has_custom_ffb():
                            # Remove ffb_predicted line for cars with custom FFB
                            if line.strip().startswith('ffb_predicted='):
                                continue
                        elif predictions:
                            # Apply predictions for cars without custom FFB
                            if line.strip().startswith('forcefeedbacksensitivitytarmac='):
                                line = f'forcefeedbacksensitivitytarmac={predictions[0]}\n'
                            elif line.strip().startswith('forcefeedbacksensitivitygravel='):
                                line = f'forcefeedbacksensitivitygravel={predictions[1]}\n'
                            elif line.strip().startswith('forcefeedbacksensitivitysnow='):
                                line = f'forcefeedbacksensitivitysnow={predictions[2]}\n'
                            # Handle ffb_predicted line
                            elif line.strip().startswith('ffb_predicted='):
                                # Keep existing ffb_predicted line
                                line = f'ffb_predicted={predictions[0]},{predictions[1]},{predictions[2]}\n\n'
                                predictions = None  # Prevent adding another one at section end
                            # Add predicted values at end of car section if not already set
                            elif line.strip() == '':
                                line = f'ffb_predicted={predictions[0]},{predictions[1]},{predictions[2]}\n\n'
                                predictions = None  # Prevent adding another one later

                    outfile.write(line)

            logger.info(f"FFB file generation complete:")
            logger.info(f"- Total cars processed: {cars_processed}")
            logger.info(f"- Cars modified with predictions: {cars_modified}")
            logger.info(f"- Output written to: {output_file}")

        except Exception as e:
            raise Exception(f"Error generating AI FFB file: {str(e)}")

    def replace_personal_ini_with_predictions(self, ai_file: str) -> None:
        """Replace personal.ini with AI-generated FFB settings after making a backup.

        Args:
            ai_file: Path to the generated AI FFB file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        dir_name = os.path.dirname(self.personal_ini)
        base_name = os.path.basename(self.personal_ini)
        name, ext = os.path.splitext(base_name)
        backup_file = os.path.join(dir_name, f"{name}_{timestamp}{ext}")

        try:
            # Make backup of current personal.ini
            logger.info(f"Creating backup of personal.ini as {backup_file}")
            shutil.copy2(self.personal_ini, backup_file)

            # Replace personal.ini with AI-generated file
            logger.info(f"Replacing personal.ini with AI-generated settings")
            shutil.move(ai_file, self.personal_ini)
            logger.info(f"Successfully updated personal.ini with AI FFB settings")
            logger.info(f"Original file backed up as: {backup_file}")

        except Exception as e:
            logger.error(f"Failed to replace personal.ini: {str(e)}")
            if os.path.exists(backup_file):
                logger.info(f"Restoring from backup {backup_file}")
                with open(backup_file, 'r', encoding='utf-8') as src, \
                     open(self.personal_ini, 'w', encoding='utf-8') as dst:
                    dst.write(src.read())
            raise

    def validate_predictions(self, models: dict) -> None:
        """Validate model predictions against known FFB settings"""
        correct = 0
        total = 0

        for car in self.cars.values():
            if car.has_custom_ffb():

                pred_tarmac, pred_gravel, pred_snow = self.predict_ffb_settings(car, models)

                # Check if predictions are within 10% of actual values
                tarmac_ok = abs(pred_tarmac - car.ffb_tarmac) <= (car.ffb_tarmac * 0.1)
                gravel_ok = abs(pred_gravel - car.ffb_gravel) <= (car.ffb_gravel * 0.1)
                snow_ok = abs(pred_snow - car.ffb_snow) <= (car.ffb_snow * 0.1)

                if tarmac_ok and gravel_ok and snow_ok:
                    correct += 1
                total += 1

                logger.debug(f"Car {car.id} predictions: "
                           f"T:{pred_tarmac}({car.ffb_tarmac}) "
                           f"G:{pred_gravel}({car.ffb_gravel}) "
                           f"S:{pred_snow}({car.ffb_snow})")

        if total > 0:
            accuracy = (correct / total) * 100
            logger.info(f"Prediction accuracy within 10%: {accuracy:.1f}% ({correct}/{total} cars)")



