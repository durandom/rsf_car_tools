import os
import argparse
from io import StringIO
from configobj import ConfigObj
from typing import List, Tuple, Dict

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
            print(f"Error parsing {file}: {str(e)}")
            exit(1)

    def _load_cars(self) -> None:
        """Load car configurations from personal.ini"""
        config = self.config_parser(self.personal_ini)

        for section_name in config:
            if not section_name.startswith('car'):
                continue

            car_id = section_name[3:]  # Remove 'car' prefix
            # Get section data and ensure string types
            section = config[section_name]
            car_data = {str(k): str(v) for k, v in section.items()}
            print(car_id, car_data)
            self.cars[car_id] = Car(car_id, car_data)

class Car:
    def __init__(self, id: str, data: Dict[str, str]):
        """Initialize car configuration

        Args:
            id (str): Car ID number
            data (Dict[str, str]): Car configuration data
        """
        self.id = id
        self.name = data.get('name', '')
        self.ffb_tarmac = int(data.get('forcefeedbacksensitivitytarmac', 0))
        self.ffb_gravel = int(data.get('forcefeedbacksensitivitygravel', 0))
        self.ffb_snow = int(data.get('forcefeedbacksensitivitysnow', 0))

def main():
    parser = argparse.ArgumentParser(description='Modify RSF power polygon settings')
    parser.add_argument('rsf_path', help='Path to RSF installation directory')

    args = parser.parse_args()

    try:
        rsf = Rsf(args.rsf_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    return 0

if __name__ == '__main__':
    exit(main())
