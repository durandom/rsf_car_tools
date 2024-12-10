import os
import pytest
from powersteering.core import PowerSteering

@pytest.fixture
def ps():
    """Create a PowerSteering instance using test fixtures"""
    fixtures_path = os.path.join(os.path.dirname(__file__), 'fixtures')
    return PowerSteering(fixtures_path)

def test_car_count(ps):
    """Test that PowerSteering correctly loads and counts cars"""
    # Count total cars (both driven and undriven)
    total_cars = len(ps.cars) + len(ps.undriven_cars)
    assert total_cars == 99, "99 cars were loaded"

    # Verify we have some driven cars
    assert len(ps.cars) == 84, "84 driven cars found"

    # Verify we have some undriven cars
    assert len(ps.undriven_cars) == 15, "15 undriven cars found"
