import os
import json
import argparse
import sys
import numpy as np
from io import StringIO
from configobj import ConfigObj
from typing import List, Tuple, Dict, Optional, TYPE_CHECKING
from loguru import logger
import plotext as plt
from rich.console import Console
from rich.table import Table

from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

def setup_logging(verbose_count: int) -> None:
    """Configure logging level based on verbosity count

    Args:
        verbose_count: Number of times --verbose flag was specified
            0 = INFO only
            1 = Add WARNING
            2 = Add DEBUG
            3 = Add everything
    """
    logger.remove()  # Remove default handler

    # Map verbosity count to log level using all loguru severity levels
    levels = {
        0: "CRITICAL",  # 50 (show only critical)
        1: "ERROR",     # 40 (show error and above)
        2: "WARNING",   # 30 (show warning and above)
        3: "SUCCESS",   # 25 (show success and above)
        4: "INFO",      # 20 (show info and above)
        5: "DEBUG",     # 10 (show debug and above)
        6: "TRACE"      # 5  (show everything)
    }
    level = levels.get(min(verbose_count, max(levels.keys())), "CRITICAL")
    logger.add(
        sys.stdout,
        colorize=True,
        level=level
    )
    logger.success(f"Set log level to {level} based on verbosity count {verbose_count}")

class Car:
    def __init__(self, id: str, data: Dict[str, str]):
        """Initialize car configuration

        Args:
            id (str): Car ID number
            data (Dict[str, str]): Car configuration data from personal.ini
        """
        self.id = id
        self.name = data.get('name', '')
        self.ffb_tarmac = int(data.get('forcefeedbacksensitivitytarmac', 0))
        self.ffb_gravel = int(data.get('forcefeedbacksensitivitygravel', 0))
        self.ffb_snow = int(data.get('forcefeedbacksensitivitysnow', 0))
        self.ffb_predicted = data.get('ffb_predicted', '').lower() == 'true'

        # These will be populated from cars.json later
        self.path = ''
        self.hash = ''
        self.carmodel_id = ''
        self.user_id = ''
        self.base_group_id = ''
        self.ngp = ''
        self.custom_setups = ''
        self.rev = ''
        self.audio = None
        self.audio_hash = ''

        # Car data attributes
        self.power = 0
        self.torque = 0
        self.drive_train = ''
        self.engine = ''
        self.transmission = ''
        self.weight = 0
        self.wdf = ''
        self.steering_wheel = 0
        self.skin = ''
        self.model = ''
        self.year = ''
        self.shifter_type = ''
        self.cluster = None  # Store which cluster this car belongs to

class Rsf:
    def __init__(self, rsf_path: str, record_html: bool = False):
        """Initialize RSF configuration handler

        Args:
            rsf_path (str): Path to RSF installation directory
            record_html (bool): Whether to record console output for HTML export
        """
        self.rsf_path = rsf_path
        self.features = ['weight', 'steering_wheel', 'drive_train']  # Default features
        self.console = Console(record=record_html)

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
        self._load_personal_ini()
        self._load_rbr_ini()
        self._load_cars_json()
        self._load_cars_data_json()
        self.drive_map = self._build_drive_map()
        self._log_cars_statistics()

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
                contents = contents.replace(';', '#')

            # Convert contents to list of strings and pass to ConfigObj
            config = ConfigObj(StringIO(contents), encoding='utf-8')
            return config

        except Exception as e:
            logger.error(f"Error parsing {file}: {str(e)}")
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

    def has_custom_ffb(self, car: Car) -> bool:
        """Check if a car has custom force feedback settings different from global defaults.

        Compares the car's individual tarmac, gravel and snow FFB settings against
        the global defaults from RichardBurnsRally.ini.

        Args:
            car: Car object to check for custom FFB settings

        Returns:
            True if the car has any FFB settings that differ from global defaults
            and were not predicted by AI, False otherwise
        """
        if car.ffb_predicted:
            return False

        return (car.ffb_tarmac != self.ffb_tarmac or
                car.ffb_gravel != self.ffb_gravel or
                car.ffb_snow != self.ffb_snow)

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

    def list_undriven_cars(self) -> None:
        """Display list of undriven cars with key details"""
        table = Table(title="Undriven Cars", show_header=True)
        table.add_column("Car Name", style="cyan")
        table.add_column("Details", style="green")

        for car_id, car in sorted(self.undriven_cars.items(), key=lambda x: x[1].name):
            table.add_row(
                car.name,
                self.format_car_details(car)
            )

        self.console.print("\n")
        self.console.print(table)
        self.console.print("\n")

    def _log_cars_statistics(self) -> None:
        """Display statistics about loaded cars"""
        # Create statistics table
        table = Table(title="Car Statistics", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Count", justify="right", style="green")

        total_cars = len(self.cars)
        ffb_cars = sum(1 for car in self.cars.values() if self.has_custom_ffb(car))
        undriven_cars = len(self.undriven_cars)

        table.add_row("Total Cars", str(total_cars))
        table.add_row("Cars with Custom FFB", str(ffb_cars))
        table.add_row("Undriven Cars", str(undriven_cars))

        self.console.print("\n")
        self.console.print(table)
        self.console.print("\n")

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
            if self.has_custom_ffb(car):
                feature_values = self._extract_feature_values(car)
                if feature_values:
                    features.append(feature_values)
                    targets.append([car.ffb_tarmac, car.ffb_gravel, car.ffb_snow])

        return np.array(features), np.array(targets)

    def create_ffb_model(self) -> Pipeline:
        """Create a polynomial regression pipeline for FFB prediction.

        Builds a scikit-learn pipeline with:
        1. StandardScaler for feature normalization
        2. PolynomialFeatures for linear feature transformation
        3. LinearRegression for prediction

        Returns:
            Pipeline: Configured scikit-learn pipeline ready for training
        """
        # Create pipeline with scaling, polynomial features, and regression
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=3)),  # Linear features only
            ('regressor', LinearRegression(fit_intercept=True))
        ])

        return model

    def train_ffb_models(self):
        """Train separate models for each surface type"""
        X, y = self.prepare_training_data()
        if len(X) == 0:
            logger.error("No valid training data found")
            return None

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Train separate model for each surface
        models = {}
        surface_names = ['tarmac', 'gravel', 'snow']

        for i, surface in enumerate(surface_names):
            model = self.create_ffb_model()
            model.fit(X_train, y_train[:, i])
            score = model.score(X_test, y_test[:, i])
            logger.info(f"{surface.title()} model R² score: {score:.3f}")
            models[surface] = model

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

            # Predict for each surface and ensure reasonable values
            predictions = []
            for surface in ['tarmac', 'gravel', 'snow']:
                pred = models[surface].predict(features)[0]
                # Clip prediction to reasonable range (50-2000)
                pred = max(50, min(2000, pred))
                predictions.append(int(round(pred)))

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

    def display_selected_sample(self, selected_cars: List[Car]) -> None:
        """Display details of selected car sample and cluster statistics

        Args:
            selected_cars: List of selected Car objects
        """
        # First display cluster statistics
        cluster_stats = Table(title="Cluster Statistics", show_header=True)
        cluster_stats.add_column("Cluster", justify="right", style="magenta")
        cluster_stats.add_column("Size", justify="right")
        cluster_stats.add_column("Avg Weight", justify="right")
        cluster_stats.add_column("Avg Steering", justify="right")
        cluster_stats.add_column("Drivetrain Distribution")

        # Calculate statistics per cluster using all valid cars
        from collections import defaultdict
        cluster_data = defaultdict(list)

        # First get all valid cars and their clusters
        features = []
        valid_cars = []
        for car in self.cars.values():
            feature_values = self._extract_feature_values(car)
            if feature_values:
                features.append(feature_values)
                valid_cars.append(car)

        # Perform clustering on all valid cars
        features = np.array(features)
        scaler = MinMaxScaler()
        features_scaled = scaler.fit_transform(features)
        n_clusters = self._find_optimal_clusters(features_scaled)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(features_scaled)

        # Assign clusters to all valid cars
        for car, cluster_id in zip(valid_cars, clusters):
            car.cluster = cluster_id
            cluster_data[cluster_id].append(car)

        for cluster_id, cars in sorted(cluster_data.items()):
            # Calculate weight range
            weights = [car.weight for car in cars]
            min_weight = min(weights)
            max_weight = max(weights)
            weight_dist = f"{min_weight}-{max_weight} kg"

            # Calculate steering range
            steering = [car.steering_wheel for car in cars]
            min_steering = min(steering)
            max_steering = max(steering)
            steering_dist = f"{min_steering}-{max_steering}°"

            # Count drivetrains
            drive_counts = defaultdict(int)
            for car in cars:
                drive_counts[car.drive_train] += 1
            drive_dist = ", ".join(f"{dt}: {count}" for dt, count in drive_counts.items())

            cluster_stats.add_row(
                str(cluster_id),
                str(len(cars)),
                weight_dist,
                steering_dist,
                drive_dist
            )

        # Add summary row with distribution ranges
        total_cars = sum(len(cars) for cars in cluster_data.values())

        # Calculate weight distribution
        all_weights = [car.weight for cars in cluster_data.values() for car in cars]
        min_weight = min(all_weights)
        max_weight = max(all_weights)
        weight_dist = f"{min_weight}-{max_weight} kg"

        # Calculate steering distribution
        all_steering = [car.steering_wheel for cars in cluster_data.values() for car in cars]
        min_steering = min(all_steering)
        max_steering = max(all_steering)
        steering_dist = f"{min_steering}-{max_steering}°"

        # Count all drivetrains
        all_drive_counts = defaultdict(int)
        for cars in cluster_data.values():
            for car in cars:
                all_drive_counts[car.drive_train] += 1
        all_drive_dist = ", ".join(f"{dt}: {count}" for dt, count in sorted(all_drive_counts.items()))

        # Add summary row with different style
        cluster_stats.add_row(
            "Total",
            str(total_cars),
            weight_dist,
            steering_dist,
            all_drive_dist,
            style="bold cyan"
        )

        self.console.print("\n")
        self.console.print(cluster_stats)
        self.console.print("\n")

        # Then display selected cars
        table = Table(title=f"Selected Training Sample ({len(cluster_data)} clusters)", show_header=True)
        table.add_column("Cluster", justify="right", style="magenta")
        table.add_column("Car", style="cyan")
        table.add_column("Weight", justify="right")
        table.add_column("Steering", justify="right")
        table.add_column("Drivetrain")
        table.add_column("FFB (T/G/S)", justify="right")

        current_cluster = None
        for car in selected_cars:
            # Check if we're starting a new cluster
            if current_cluster != car.cluster:
                current_cluster = car.cluster
                if current_cluster is not None:
                    # Alternate between dark grey and default background
                    row_style = "on grey30" if current_cluster % 2 == 0 else None

            table.add_row(
                str(car.cluster),
                f"{car.id} - {car.name}",
                f"{car.weight}",
                f"{car.steering_wheel}°",
                car.drive_train,
                f"{car.ffb_tarmac}/{car.ffb_gravel}/{car.ffb_snow}",
                style=row_style
            )

        self.console.print("\n")
        self.console.print(table)
        self.console.print("\n")

    def set_features(self, features: List[str]) -> None:
        """Set the features to use for training and prediction

        Args:
            features: List of feature names (must be attributes of Car class)
        """
        self.features = features
        logger.info(f"Set features to: {features}")

    def display_ffb_generation_results(self, cars_with_predictions: List[Tuple[Car, Tuple[int, int, int]]]) -> None:
        """Display table of FFB generation results

        Args:
            cars_with_predictions: List of (Car, (tarmac, gravel, snow)) tuples
        """
        table = Table(title="FFB Generation Results", show_header=True)
        table.add_column("Car", style="cyan")
        table.add_column("Weight", justify="right")
        table.add_column("Steering", justify="right")
        table.add_column("Drivetrain")
        table.add_column("Current FFB (T/G/S)", justify="right")
        table.add_column("Predicted FFB (T/G/S)", justify="right")
        table.add_column("Status")

        for car, predictions in cars_with_predictions:
            # Determine row style based on whether car has custom FFB
            row_style = "on red" if self.has_custom_ffb(car) else None

            current_ffb = f"{car.ffb_tarmac}/{car.ffb_gravel}/{car.ffb_snow}"
            predicted_ffb = f"{predictions[0]}/{predictions[1]}/{predictions[2]}"
            status = "Skipped - Custom FFB" if self.has_custom_ffb(car) else "Updated"

            table.add_row(
                f"{car.id} - {car.name}",
                f"{car.weight}",
                f"{car.steering_wheel}°",
                car.drive_train,
                current_ffb,
                predicted_ffb,
                status,
                style=row_style
            )

        self.console.print("\n")
        self.console.print(table)
        self.console.print("\n")

    def generate_ai_ffb_file(self, models: dict, output_file: str) -> None:
        """Generate a new personal.ini file with AI-predicted FFB settings

        Args:
            models: Dictionary of trained models for each surface
            output_file: Path to output file
        """
        if not models:
            logger.error("No trained models available")
            return

        current_car_id = None
        current_car = None
        predictions = None
        cars_with_predictions = []

        logger.info(f"Starting FFB file generation: reading from {self.personal_ini}")
        logger.info(f"Writing predictions to {output_file}")

        cars_processed = 0
        cars_modified = 0

        try:
            with open(self.personal_ini, 'r', encoding='utf-8') as infile, \
                 open(output_file, 'w', encoding='utf-8') as outfile:

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

                            if current_car:
                                predictions = self.predict_ffb_settings(current_car, models)
                                cars_with_predictions.append((current_car, predictions))
                                if not self.has_custom_ffb(current_car):
                                    cars_modified += 1
                                    logger.debug(f"Set predicted FFB for car {current_car_id}: "
                                               f"T:{predictions[0]} G:{predictions[1]} S:{predictions[2]}")
                            else:
                                logger.warning(f"Car {current_car_id} found in personal.ini but not in cars.json")
                                predictions = None

                    # Check for FFB settings if we have predictions for a car section
                    if predictions and current_car and not self.has_custom_ffb(current_car):
                        if line.strip().startswith('forcefeedbacksensitivitytarmac='):
                            line = f'forcefeedbacksensitivitytarmac={predictions[0]}\n'
                        elif line.strip().startswith('forcefeedbacksensitivitygravel='):
                            line = f'forcefeedbacksensitivitygravel={predictions[1]}\n'
                        elif line.strip().startswith('forcefeedbacksensitivitysnow='):
                            line = f'forcefeedbacksensitivitysnow={predictions[2]}\n'
                        # Only add ffb_predicted in car sections
                        elif line.strip() == '' and current_car_id:  # At end of car section
                            line = 'ffb_predicted=true\n\n'

                    outfile.write(line)

            logger.info(f"FFB file generation complete:")
            logger.info(f"- Total cars processed: {cars_processed}")
            logger.info(f"- Cars modified with predictions: {cars_modified}")
            logger.info(f"- Output written to: {output_file}")

            # Display results table
            self.display_ffb_generation_results(cars_with_predictions)

        except Exception as e:
            raise Exception(f"Error generating AI FFB file: {str(e)}")

    def validate_predictions(self, models: dict) -> None:
        """Validate model predictions against known FFB settings"""
        correct = 0
        total = 0

        for car in self.cars.values():
            if self.has_custom_ffb(car):

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

    def _plot_numeric_histogram(self, values, title, xlabel):
        """Plot histogram for numeric data"""
        if not values:
            logger.error(f"No valid data found for {title}")
            return

        plt.clear_figure()
        plt.hist(values, bins=10)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Number of Cars")
        plt.show()

    def _plot_categorical_distribution(self, values, title, xlabel):
        """Plot bar chart for categorical data"""
        if not values:
            logger.error(f"No valid data found for {title}")
            return

        # Count occurrences of each category
        from collections import Counter
        counts = Counter(values)
        categories = list(counts.keys())
        frequencies = list(counts.values())

        plt.clear_figure()
        plt.bar(categories, frequencies)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Number of Cars")
        plt.show()

    def plot_weight_stats(self):
        """Plot histogram of car weights"""
        weights = []
        for car in self.cars.values():
            try:
                if car.weight:
                    weights.append(car.weight)
            except ValueError:
                continue
        self._plot_numeric_histogram(weights, "Car Weight Distribution", "Weight (kg)")

    def plot_drivetrain_stats(self):
        """Plot distribution of car drivetrains"""
        drivetrains = [car.drive_train for car in self.cars.values() if car.drive_train]
        self._plot_categorical_distribution(drivetrains, "Car Drivetrain Distribution", "Drivetrain Type")

    def plot_steering_stats(self):
        """Plot histogram of steering wheel angles"""
        steering_angles = [car.steering_wheel for car in self.cars.values() if car.steering_wheel is not None]
        self._plot_numeric_histogram(steering_angles, "Steering Wheel Angle Distribution", "Angle (degrees)")



def main():
    parser = argparse.ArgumentParser(description='Modify RSF power polygon settings')
    parser.add_argument('rsf_path', help='Path to RSF installation directory')
    parser.add_argument('--verbose', '-v', action='count', default=0, help='Increase verbosity level (can be used multiple times)')
    parser.add_argument('--stats', help='Comma-separated list of statistics to plot (weight)')
    parser.add_argument('--train', action='store_true', help='Train FFB prediction models')
    parser.add_argument('--validate', action='store_true', help='Validate FFB predictions')
    parser.add_argument('--undriven', action='store_true', help='List undriven cars')
    parser.add_argument('--select-sample', type=int, nargs='?', const=3, default=None, metavar='N',
                       help='Select N cars from each cluster for training sample (default: 3)')
    parser.add_argument('--html', type=str, help='Save console output to HTML file')
    parser.add_argument('--generate', action='store_true', help='Generate rallysimfans_personal_ai.ini with AI-predicted FFB settings')

    args = parser.parse_args()
    setup_logging(args.verbose)

    try:
        rsf = Rsf(args.rsf_path, record_html=bool(args.html))

        if args.stats:
            stats_list = [s.strip().lower() for s in args.stats.split(',')]
            if 'weight' in stats_list:
                rsf.plot_weight_stats()
            if 'drivetrain' in stats_list:
                rsf.plot_drivetrain_stats()
            if 'steering' in stats_list:
                rsf.plot_steering_stats()

        if args.undriven:
            rsf.list_undriven_cars()

        if args.select_sample:
            selected = rsf.select_training_sample(args.select_sample)
            rsf.display_selected_sample(selected)

        if args.train or args.validate or args.generate:
            models = rsf.train_ffb_models()
            if models:
                if args.validate:
                    rsf.validate_predictions(models)
                if args.generate:
                    output_file = os.path.join(args.rsf_path, 'rallysimfans_personal_ai.ini')
                    rsf.generate_ai_ffb_file(models, output_file)

    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        return 1
    finally:
        # Save HTML output if requested
        if args.html:
            rsf.console.save_html(args.html)

    return 0

if __name__ == '__main__':
    exit(main())
