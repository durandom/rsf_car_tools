import os
import json
import argparse
import numpy as np
from io import StringIO
from configobj import ConfigObj
from typing import List, Tuple, Dict, Optional, TYPE_CHECKING
from loguru import logger
import plotext as plt

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

def setup_logging(verbose: bool) -> None:
    """Configure logging level based on verbosity"""
    logger.remove()  # Remove default handler
    level = "DEBUG" if verbose else "INFO"
    logger.add(sink=lambda msg: print(msg), level=level)

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
        self.power = ''
        self.torque = ''
        self.drive_train = ''
        self.engine = ''
        self.transmission = ''
        self.weight = ''
        self.wdf = ''
        self.steering_wheel = 0.0
        self.skin = ''
        self.model = ''
        self.year = ''
        self.shifter_type = ''
class Rsf:
    def __init__(self, rsf_path: str):
        """Initialize RSF configuration handler

        Args:
            rsf_path (str): Path to RSF installation directory
        """
        self.rsf_path = rsf_path

        # Define required files
        self.personal_ini = os.path.join(rsf_path, 'rallysimfans_personal.ini')
        self.rbr_ini = os.path.join(rsf_path, 'RichardBurnsRally.ini')
        self.cars_json = os.path.join(rsf_path, 'rsfdata', 'cache', 'cars.json')
        self.cars_data_json = os.path.join(rsf_path, 'rsfdata', 'cache', 'cars_data.json')

        self._validate_files()
        self.cars: Dict[str, Car] = {}
        # Global FFB settings from RBR
        self.ffb_tarmac = 0
        self.ffb_gravel = 0
        self.ffb_snow = 0
        self._load_personal_ini()
        self._load_rbr_ini()
        self._load_cars_json()
        self._load_cars_data_json()
        self._log_cars_statistics()

    def _validate_files(self) -> None:
        """Validate all required files exist"""
        missing_files = []

        for filepath in [self.personal_ini, self.rbr_ini, self.cars_json, self.cars_data_json]:
            if not os.path.exists(filepath):
                missing_files.append(filepath)

        if missing_files:
            raise FileNotFoundError(
                "Missing required files:\n" +
                "\n".join(f"- {f}" for f in missing_files)
            )

    def config_parser(self, file):
        """Parse ini file handling comments and duplicates like rbr_pacenote_plugin"""
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

    def _load_cars_json(self) -> None:
        """Load cars.json and add data to existing Car objects"""
        try:
            with open(self.cars_json, 'r', encoding='utf-8') as f:
                cars_json = json.load(f)
                for car_json in cars_json:
                    car_id = car_json['id']
                    if car_id in self.cars:
                        car = self.cars[car_id]
                        # Add json attributes to existing Car object
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
                        logger.debug(f"Added JSON data to car {car_id}")
        except Exception as e:
            logger.error(f"Error loading cars.json: {str(e)}")

    def has_custom_ffb(self, car: Car) -> bool:
        """Check if a car has custom FFB settings different from global defaults

        Args:
            car (Car): Car object to check

        Returns:
            bool: True if car has custom FFB settings
        """
        return (car.ffb_tarmac != self.ffb_tarmac or
                car.ffb_gravel != self.ffb_gravel or
                car.ffb_snow != self.ffb_snow)

    def _log_cars_statistics(self) -> None:
        """Log statistics about loaded cars"""
        total_cars = len(self.cars)
        ffb_cars = sum(1 for car in self.cars.values() if self.has_custom_ffb(car))
        logger.info(f"Loaded {total_cars} cars total, {ffb_cars} have custom FFB settings")

    def prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features and targets from cars with custom FFB settings"""
        features = []
        targets = []

        for car in self.cars.values():
            # Only use cars with custom FFB settings
            if self.has_custom_ffb(car):

                # Extract and normalize features
                try:
                    weight = float(car.weight.lower().replace('kg', '').strip())
                    steering = float(car.steering_wheel)
                    # Encode drivetrain (RWD=1, FWD=2, AWD=3)
                    drive_map = {'RWD': 1, 'FWD': 2, 'AWD': 3}
                    drivetrain = drive_map.get(car.drive_train.upper(), 0)

                    if weight > 0 and steering > 0 and drivetrain > 0:
                        features.append([weight, steering, drivetrain])
                        targets.append([car.ffb_tarmac, car.ffb_gravel, car.ffb_snow])
                except (ValueError, AttributeError):
                    continue

        return np.array(features), np.array(targets)

    def create_ffb_model(self):
        """Create and train polynomial regression model"""
        # Create pipeline with scaling, polynomial features, and regression
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=2)),
            ('regressor', LinearRegression())
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

    def predict_ffb_settings(self, car: Car, models: dict) -> Tuple[int, int, int]:
        """Predict FFB settings for a car using trained models"""
        try:
            # Extract features
            weight = float(car.weight.lower().replace('kg', '').strip())
            steering = float(car.steering_wheel)
            drive_map = {'RWD': 1, 'FWD': 2, 'AWD': 3}
            drivetrain = drive_map.get(car.drive_train.upper(), 0)

            if weight <= 0 or steering <= 0 or drivetrain <= 0:
                return (self.ffb_tarmac, self.ffb_gravel, self.ffb_snow)

            features = np.array([[weight, steering, drivetrain]])

            # Predict for each surface
            ffb_tarmac = int(round(models['tarmac'].predict(features)[0]))
            ffb_gravel = int(round(models['gravel'].predict(features)[0]))
            ffb_snow = int(round(models['snow'].predict(features)[0]))

            # Clamp values to valid range (0-200)
            ffb_tarmac = max(0, min(200, ffb_tarmac))
            ffb_gravel = max(0, min(200, ffb_gravel))
            ffb_snow = max(0, min(200, ffb_snow))

            return (ffb_tarmac, ffb_gravel, ffb_snow)

        except (ValueError, AttributeError) as e:
            logger.warning(f"Could not predict FFB settings for car {car.id}: {str(e)}")
            return (self.ffb_tarmac, self.ffb_gravel, self.ffb_snow)

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
                    weight = float(car.weight.lower().replace('kg', '').strip())
                    weights.append(weight)
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
                            car.power = car_data.get('power', '')
                            car.torque = car_data.get('torque', '')
                            car.drive_train = car_data.get('drive_train', '')
                            car.engine = car_data.get('engine', '')
                            car.transmission = car_data.get('transmission', '')
                            car.weight = car_data.get('weight', '')
                            car.wdf = car_data.get('wdf', '')
                            # Parse steering wheel angle to numeric value
                            steering_wheel = car_data.get('steering_wheel', '')
                            try:
                                if steering_wheel:
                                    car.steering_wheel = float(steering_wheel.replace('°', '').strip())
                                else:
                                    car.steering_wheel = 0
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



def main():
    parser = argparse.ArgumentParser(description='Modify RSF power polygon settings')
    parser.add_argument('rsf_path', help='Path to RSF installation directory')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable debug logging')
    parser.add_argument('--stats', help='Comma-separated list of statistics to plot (weight)')
    parser.add_argument('--train', action='store_true', help='Train FFB prediction models')
    parser.add_argument('--validate', action='store_true', help='Validate FFB predictions')

    args = parser.parse_args()
    setup_logging(args.verbose)

    try:
        rsf = Rsf(args.rsf_path)

        if args.stats:
            stats_list = [s.strip().lower() for s in args.stats.split(',')]
            if 'weight' in stats_list:
                rsf.plot_weight_stats()
            if 'drivetrain' in stats_list:
                rsf.plot_drivetrain_stats()
            if 'steering' in stats_list:
                rsf.plot_steering_stats()

        if args.train or args.validate:
            models = rsf.train_ffb_models()
            if models and args.validate:
                rsf.validate_predictions(models)

    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        return 1

    return 0

if __name__ == '__main__':
    exit(main())
