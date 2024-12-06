import argparse
import os
from loguru import logger

from .core import PowerSteering
from .utils import setup_logging

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

    args = parser.parse_args()
    setup_logging(args.verbose)

    try:
        ps = PowerSteering(args.rsf_path, record_html=bool(args.html))

        if args.stats:
            stats_list = [s.strip().lower() for s in args.stats.split(',')]
            if 'weight' in stats_list:
                ps.plot_weight_stats()
            if 'drivetrain' in stats_list:
                ps.plot_drivetrain_stats() 
            if 'steering' in stats_list:
                ps.plot_steering_stats()

        if args.undriven:
            ps.list_undriven_cars()

        if args.select_sample:
            selected = ps.select_training_sample(args.select_sample)
            ps.display_selected_sample(selected)

        if args.train or args.validate or args.generate:
            models = ps.train_ffb_models()
            if models:
                if args.validate:
                    ps.validate_predictions(models)
                if args.generate:
                    output_file = os.path.join(args.rsf_path, 'rallysimfans_personal_ai.ini')
                    ps.generate_ai_ffb_file(models, output_file)

    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        return 1
    finally:
        if args.html:
            ps.save_html(args.html)

    return 0

if __name__ == '__main__':
    exit(main())
