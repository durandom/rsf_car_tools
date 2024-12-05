import os
import json
import argparse
from io import StringIO
from configobj import ConfigObj
from typing import List, Tuple, Dict, Optional
from loguru import logger

def setup_logging(verbose: bool) -> None:
    """Configure logging level based on verbosity"""
    logger.remove()  # Remove default handler
    level = "DEBUG" if verbose else "INFO"
    logger.add(sink=lambda msg: print(msg), level=level)

class Rsf:
    def __init__(self, rsf_path: str):
        """Initialize RSF configuration handler

        Args:
            rsf_path (str): Path to RSF installation directory
        """
        self.rsf_path = rsf_path

        # Define required files
        self.personal_ini = os.path.join(rsf_path, 'rallysimfans_personal.ini')
        self.cars_json = os.path.join(rsf_path, 'rsfdata', 'cache', 'cars.json')
        self.cars_data_json = os.path.join(rsf_path, 'rsfdata', 'cache', 'cars_data.json')

        self._validate_files()
        self.cars: Dict[str, Car] = {}
        self._load_cars()
        self._augment_cars_with_json()

    def _validate_files(self) -> None:
        """Validate all required files exist"""
        missing_files = []

        for filepath in [self.personal_ini, self.cars_json, self.cars_data_json]:
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

    def _augment_cars_with_json(self) -> None:
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

    def _load_cars(self) -> None:
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

def main():
    parser = argparse.ArgumentParser(description='Modify RSF power polygon settings')
    parser.add_argument('rsf_path', help='Path to RSF installation directory')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable debug logging')

    args = parser.parse_args()
    setup_logging(args.verbose)

    try:
        rsf = Rsf(args.rsf_path)
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        return 1

    return 0

if __name__ == '__main__':
    exit(main())
