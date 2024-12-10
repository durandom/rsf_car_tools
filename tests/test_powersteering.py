import os
import pytest
from powersteering.core import PowerSteering

@pytest.fixture
def ps():
    """Create a PowerSteering instance using test fixtures"""
    fixtures_path = os.path.join(os.path.dirname(__file__), 'fixtures')
    return PowerSteering(fixtures_path, global_ffb=True)

def test_car_count(ps):
    """Test that PowerSteering correctly loads and counts cars"""
    # Count total cars (both driven and undriven)
    total_cars = len(ps.cars) + len(ps.undriven_cars)
    assert total_cars == 99, "99 cars were loaded"

    # Verify we have some driven cars
    assert len(ps.cars) == 84, "84 driven cars found"

    # Verify we have some undriven cars
    assert len(ps.undriven_cars) == 15, "15 undriven cars found"

    # Number of cars with custom FFB settings
    ffb_cars = sum(1 for car in ps.cars.values() if car.has_custom_ffb())

    assert ffb_cars == 10, "10 cars have custom FFB settings"

    # Number of cars with predicted FFB settings
    predicted_ffb_cars = sum(1 for car in ps.cars.values() if car.ffb_predicted)
    assert predicted_ffb_cars == 74, "74 cars have predicted FFB settings"

def test_prediction_determinism(ps):
    """Test that FFB predictions are deterministic"""
    # Train models twice
    models1 = ps.train_ffb_models()
    models2 = ps.train_ffb_models()

    # Get predictions from both models for all cars
    predictions1 = ps.predict_all_ffb_settings(models1)
    predictions2 = ps.predict_all_ffb_settings(models2)

    # Compare predictions
    for (car1, pred1), (car2, pred2) in zip(predictions1, predictions2):
        assert car1.id == car2.id, "Cars should be in same order"
        assert pred1 == pred2, f"Predictions differ for car {car1.id}: {pred1} != {pred2}"
