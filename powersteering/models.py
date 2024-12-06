from typing import Dict

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
