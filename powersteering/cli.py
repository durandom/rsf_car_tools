import argparse
import os
import sys
from loguru import logger

from powersteering.core import PowerSteering
from powersteering.utils import setup_logging
from powersteering.renderer import ConsoleRenderer
from powersteering.tui import PowerSteeringApp

def main():
    parser = argparse.ArgumentParser(description='Modify RSF power steering settings')
    parser.add_argument('rsf_path', help='Path to RSF installation directory')
    parser.add_argument('--verbose', '-v', action='count', default=0,
                       help='Increase verbosity level (can be used multiple times)')
    parser.add_argument('--stats', help='Comma-separated list of statistics to plot (weight)')
    parser.add_argument('--train', action='store_true', help='Train FFB prediction models')
    parser.add_argument('--validate', action='store_true', help='Validate FFB predictions')
    parser.add_argument('--undriven', action='store_true', help='List undriven cars')
    parser.add_argument('--select-sample', type=int, nargs='?', const=3, default=None,
                       metavar='N', help='Select N cars from each cluster for training sample (default: 3)')
    parser.add_argument('--html', type=str, help='Save console output to HTML file')
    parser.add_argument('--generate', action='store_true',
                       help='Generate rallysimfans_personal_ai.ini with AI-predicted FFB settings')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show predicted FFB settings without writing to file')
    parser.add_argument('--tui', type=lambda x: x.lower() == 'true', default=True,
                       help='Launch the text user interface (default: True)')

    args = parser.parse_args()
    setup_logging(args.verbose)

    renderer = ConsoleRenderer(record_html=bool(args.html))

    if args.tui:
        app = PowerSteeringApp(args.rsf_path)
        app.run()
        return 0

    try:
        ps = PowerSteering(args.rsf_path)

        # Display initial statistics
        renderer.display_car_statistics(ps.cars, ps.undriven_cars, ps.has_custom_ffb)

        if args.stats:
            stats_list = [s.strip().lower() for s in args.stats.split(',')]
            if 'weight' in stats_list:
                renderer.plot_weight_stats(ps.cars)
            if 'drivetrain' in stats_list:
                renderer.plot_drivetrain_stats(ps.cars)
            if 'steering' in stats_list:
                renderer.plot_steering_stats(ps.cars)

        if args.undriven:
            renderer.display_undriven_cars(ps.undriven_cars, ps.format_car_details)

        if args.select_sample:
            selected = ps.select_training_sample(args.select_sample)
            cluster_data = ps.get_cluster_data(selected)
            renderer.display_selected_sample(selected, cluster_data)

        if args.train or args.validate or args.generate:
            models = ps.train_ffb_models()
            if models:
                if args.validate:
                    ps.validate_predictions(models)
                if args.generate or args.dry_run:
                    cars_with_predictions = ps.predict_all_ffb_settings(models)
                    renderer.display_ffb_generation_results(cars_with_predictions, ps.has_custom_ffb)

                    if args.generate and not args.dry_run:
                        output_file = os.path.join(args.rsf_path, 'rallysimfans_personal_ai.ini')
                        ps.write_ai_ffb_file(cars_with_predictions, output_file)

    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        return 1
    finally:
        if args.html:
            renderer.save_html(args.html)

    return 0

if __name__ == '__main__':
    sys.exit(main())
