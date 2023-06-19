import argparse

from evaluation.evaluators.console.console_evaluator import ConsoleEvaluator

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate CoNN algorithms using a config file.')
    parser.add_argument('config_file', type=str, help='Path to the configuration file.')
    args = parser.parse_args()
    ConsoleEvaluator(config_file_path=args.config_file)
