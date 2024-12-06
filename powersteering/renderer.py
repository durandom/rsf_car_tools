from rich.console import Console
from rich.table import Table
from typing import List, Tuple, Dict
from loguru import logger
from collections import Counter
import plotext as plt
from .models import Car

class ConsoleRenderer:
    """Handles all console output rendering using Rich"""

    def __init__(self, record_html: bool = False, quiet: bool = False):
        """Initialize console renderer

        Args:
            record_html: Whether to record output for HTML export
            quiet: Whether to suppress direct console output
        """
        self.console = Console(record=record_html, quiet=quiet)

    def save_html(self, filename: str) -> None:
        """Save console output to HTML file"""
        self.console.save_html(filename)

    def create_stats_table(self, cars: Dict[str, Car], undriven_cars: Dict[str, Car],
                          has_custom_ffb_func) -> Table:
        """Create summary statistics table"""
        table = Table(title="Car Statistics", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Count", justify="right", style="green")

        total_cars = len(cars)
        ffb_cars = sum(1 for car in cars.values() if has_custom_ffb_func(car))
        undriven_cars_count = len(undriven_cars)

        table.add_row("Total Cars", str(total_cars))
        table.add_row("Cars with Custom FFB", str(ffb_cars))
        table.add_row("Undriven Cars", str(undriven_cars_count))

        return table

    def create_undriven_table(self, undriven_cars: Dict[str, Car]) -> Table:
        """Create table of undriven cars"""
        table = Table(title="Undriven Cars", show_header=True)
        table.add_column("Car", style="cyan")
        table.add_column("Weight", justify="right")
        table.add_column("Steering", justify="right")
        table.add_column("Drivetrain")

        for car in sorted(undriven_cars.values(), key=lambda x: x.name):
            table.add_row(
                f"{car.id} - {car.name}",
                str(car.weight),
                f"{car.steering_wheel}°",
                car.drive_train
            )

        return table

    def create_custom_ffb_table(self, cars: Dict[str, Car], has_custom_ffb_func) -> Table:
        """Create table of cars with custom FFB settings"""
        table = Table(title="Cars with Custom FFB Settings", show_header=True)
        table.add_column("Car", style="cyan")
        table.add_column("Weight", justify="right")
        table.add_column("Steering", justify="right")
        table.add_column("Drivetrain")
        table.add_column("FFB Settings (T/G/S)", justify="right")
        table.add_column("Global FFB", justify="right")

        for car in sorted(cars.values(), key=lambda x: x.name):
            if has_custom_ffb_func(car):
                table.add_row(
                    f"{car.id} - {car.name}",
                    str(car.weight),
                    f"{car.steering_wheel}°",
                    car.drive_train,
                    f"{car.ffb_tarmac}/{car.ffb_gravel}/{car.ffb_snow}",
                    f"{car.ffb_tarmac}/{car.ffb_gravel}/{car.ffb_snow}"
                )

        return table

    def display_car_statistics(self, cars: Dict[str, Car], undriven_cars: Dict[str, Car],
                             has_custom_ffb_func) -> None:
        """Display statistics about loaded cars"""
        self.console.print("\n")
        self.console.print(self.create_stats_table(cars, undriven_cars, has_custom_ffb_func))
        self.console.print("\n")
        self.console.print(self.create_undriven_table(undriven_cars))
        self.console.print("\n")
        self.console.print(self.create_custom_ffb_table(cars, has_custom_ffb_func))
        self.console.print("\n")

    def display_undriven_cars(self, undriven_cars: Dict[str, Car], format_car_details_func) -> None:
        """Display list of undriven cars with key details"""
        table = self.create_undriven_table(undriven_cars)
        self.console.print("\n")
        self.console.print(table)
        self.console.print("\n")

    def create_cluster_stats_table(self, cluster_data: Dict) -> Table:
        """Create table showing cluster statistics"""
        from collections import defaultdict

        table = Table(title="Cluster Statistics", show_header=True)
        table.add_column("Cluster", justify="right", style="magenta")
        table.add_column("Size", justify="right")
        table.add_column("Avg Weight", justify="right")
        table.add_column("Avg Steering", justify="right")
        table.add_column("Drivetrain Distribution")

        # Add per-cluster statistics
        for cluster_id, cars in sorted(cluster_data.items()):
            weights = [car.weight for car in cars]
            weight_dist = f"{min(weights)}-{max(weights)} kg"

            steering = [car.steering_wheel for car in cars]
            steering_dist = f"{min(steering)}-{max(steering)}°"

            drive_counts = defaultdict(int)
            for car in cars:
                drive_counts[car.drive_train] += 1
            drive_dist = ", ".join(f"{dt}: {count}" for dt, count in drive_counts.items())

            table.add_row(
                str(cluster_id),
                str(len(cars)),
                weight_dist,
                steering_dist,
                drive_dist
            )

        # Add summary row
        total_cars = sum(len(cars) for cars in cluster_data.values())
        all_weights = [car.weight for cars in cluster_data.values() for car in cars]
        all_steering = [car.steering_wheel for cars in cluster_data.values() for car in cars]

        all_drive_counts = defaultdict(int)
        for cars in cluster_data.values():
            for car in cars:
                all_drive_counts[car.drive_train] += 1
        all_drive_dist = ", ".join(f"{dt}: {count}" for dt, count in sorted(all_drive_counts.items()))

        table.add_row(
            "Total",
            str(total_cars),
            f"{min(all_weights)}-{max(all_weights)} kg",
            f"{min(all_steering)}-{max(all_steering)}°",
            all_drive_dist,
            style="bold cyan"
        )

        return table

    def create_selected_cars_table(self, selected_cars: List[Car], num_clusters: int) -> Table:
        """Create table showing selected car sample"""
        table = Table(title=f"Selected Training Sample ({num_clusters} clusters)", show_header=True)
        table.add_column("Cluster", justify="right", style="magenta")
        table.add_column("Car", style="cyan")
        table.add_column("Weight", justify="right")
        table.add_column("Steering", justify="right")
        table.add_column("Drivetrain")
        table.add_column("FFB (T/G/S)", justify="right")

        current_cluster = None
        for car in selected_cars:
            if current_cluster != car.cluster:
                current_cluster = car.cluster

            row_style = "on grey30" if isinstance(current_cluster, int) and current_cluster % 2 == 0 else None

            table.add_row(
                str(car.cluster),
                f"{car.id} - {car.name}",
                f"{car.weight}",
                f"{car.steering_wheel}°",
                car.drive_train,
                f"{car.ffb_tarmac}/{car.ffb_gravel}/{car.ffb_snow}",
                style=row_style
            )

        return table

    def display_selected_sample(self, selected_cars: List[Car], cluster_data: Dict) -> None:
        """Display details of selected car sample and cluster statistics"""
        self.console.print("\n")
        self.console.print(self.create_cluster_stats_table(cluster_data))
        self.console.print("\n")
        self.console.print(self.create_selected_cars_table(selected_cars, len(cluster_data)))
        self.console.print("\n")

    def create_ffb_results_table(self, cars_with_predictions: List[Tuple[Car, Tuple[int, int, int]]],
                                has_custom_ffb_func) -> Table:
        """Create table showing FFB generation results"""
        table = Table(title="FFB Generation Results", show_header=True)
        table.add_column("Car", style="cyan")
        table.add_column("Weight", justify="right")
        table.add_column("Steering", justify="right")
        table.add_column("Drivetrain")
        table.add_column("Current FFB (T/G/S)", justify="right")
        table.add_column("Predicted FFB (T/G/S)", justify="right")
        table.add_column("Status")

        for car, predictions in cars_with_predictions:
            row_style = "on red" if has_custom_ffb_func(car) else None

            current_ffb = f"{car.ffb_tarmac}/{car.ffb_gravel}/{car.ffb_snow}"
            predicted_ffb = f"{predictions[0]}/{predictions[1]}/{predictions[2]}"
            status = "Skipped - Custom FFB" if has_custom_ffb_func(car) else "Updated"

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

        return table

    def display_ffb_generation_results(self, cars_with_predictions: List[Tuple[Car, Tuple[int, int, int]]],
                                     has_custom_ffb_func) -> None:
        """Display table of FFB generation results"""
        self.console.print("\n")
        self.console.print(self.create_ffb_results_table(cars_with_predictions, has_custom_ffb_func))
        self.console.print("\n")

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

        counts = Counter(values)
        categories = list(counts.keys())
        frequencies = list(counts.values())

        plt.clear_figure()
        plt.bar(categories, frequencies)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Number of Cars")
        plt.show()

    def plot_weight_stats(self, cars: Dict[str, Car]):
        """Plot histogram of car weights"""
        weights = []
        for car in cars.values():
            try:
                if car.weight:
                    weights.append(car.weight)
            except ValueError:
                continue
        self._plot_numeric_histogram(weights, "Car Weight Distribution", "Weight (kg)")

    def plot_drivetrain_stats(self, cars: Dict[str, Car]):
        """Plot distribution of car drivetrains"""
        drivetrains = [car.drive_train for car in cars.values() if car.drive_train]
        self._plot_categorical_distribution(drivetrains, "Car Drivetrain Distribution", "Drivetrain Type")

    def plot_steering_stats(self, cars: Dict[str, Car]):
        """Plot histogram of steering wheel angles"""
        steering_angles = [car.steering_wheel for car in cars.values() if car.steering_wheel is not None]
        self._plot_numeric_histogram(steering_angles, "Steering Wheel Angle Distribution", "Angle (degrees)")
