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
        self.ffb_tarmac = int(data.get('forcefeedbacksensitivitytarmac', 0) or 0)
        self.ffb_gravel = int(data.get('forcefeedbacksensitivitygravel', 0) or 0)
        self.ffb_snow = int(data.get('forcefeedbacksensitivitysnow', 0) or 0)

        # Parse predicted FFB values if present
        ffb_predicted = data.get('ffb_predicted', '')
        if ffb_predicted:
            try:
                predicted_values = [int(x.strip()) for x in ffb_predicted.split(',')]
                if len(predicted_values) == 3:
                    self.ffb_tarmac_predicted = predicted_values[0]
                    self.ffb_gravel_predicted = predicted_values[1]
                    self.ffb_snow_predicted = predicted_values[2]
                    self.ffb_predicted = True
                else:
                    self.ffb_predicted = False
            except ValueError:
                self.ffb_predicted = False
        else:
            self.ffb_predicted = False

        # Initialize predicted values to 0 if not set above
        if not hasattr(self, 'ffb_tarmac_predicted'):
            self.ffb_tarmac_predicted = 0
            self.ffb_gravel_predicted = 0
            self.ffb_snow_predicted = 0

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
