import time
import argparse
from matplotlib import pyplot as plt
import yaml
from balancer.model import *
from .data_interface import load_cluster_from_file, perform_migration, save_cluster_to_file, load_cluster_info
from .load_balancer import balance_cluster
from .util import Config
import sys
import logging

def run(args):
    config = Config.getInstance()
    config.load_config(args.config)

    log_level = logging.getLevelName(args.log_level) if args.log_level else logging.INFO

    setup_migration_logger()

    logging.basicConfig(filename="logs/app.log", level=log_level,
                    format="%(asctime)s %(levelname)s %(message)s")

    logging.info("Application started")

    schedule = args.schedule
    #config.test = args.test
    config.config["test"] = True
    file = args.file

    config = Config.getInstance()

    if file:
        cluster = load_cluster_from_file(file)
    else:
        cluster = load_cluster_info()

    if args.save:
        save_cluster_to_file(cluster, args.save)

    if schedule and schedule > 0:
        while True:
            print("Started cluster balancing.")
            logging.info("Started cluster balancing.")
            balance_cluster(cluster)
            time.sleep(schedule * 60)
    else:
        # Execute only once
        print("Started cluster balancing. Check logs for further info.")
        logging.info("Started cluster balancing")
        balance_cluster(cluster)

def parse_args(args):
    parser = argparse.ArgumentParser(description='Proxmox Load Balancer')
    subs = parser.add_subparsers(dest='command')

    # run sub-command
    run_parser = subs.add_parser('run', help='Run the load balancer')
    run_parser.add_argument('--schedule', type=int, help='Interval in minutes for running the load balancer (default: run only once)')
    run_parser.add_argument('--test', action='store_true', help='Run in test mode')
    run_parser.add_argument('--file', type=str, help='Path to data file')
    run_parser.add_argument('--config', type=str, help='Path to config file', required=True)
    run_parser.add_argument('--save', type=str, help='Save cluster data to file')
    run_parser.add_argument('--log-level', type=str, help='Log level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO')

    # history sub-command
    history_parser = subs.add_parser('history', help='Show history')
    history_parser.add_argument('duration', type=str, help='Duration for which history logs are to be displayed. Format: Xd for X days, Xw for X weeks, Xh for X hours')

    return parser.parse_args(args)

def print_history_logs(time_delta):
    logger = logging.getLogger("history")

    handler = logging.FileHandler('logs/history.log')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    with open(handler.baseFilename, 'r') as f:
        logs = f.readlines()
    
    # Get current time
    current_time = datetime.now()

    # Get the specified time delta in seconds
    if 'd' in time_delta:
        days = int(time_delta[:-1])
        time_delta = days * 24 * 60 * 60
    elif 'w' in time_delta:
        weeks = int(time_delta[:-1])
        time_delta = weeks * 7 * 24 * 60 * 60
    elif 'h' in time_delta:
        hours = int(time_delta[:-1])
        time_delta = hours * 60 * 60
    else:
        print("Invalid time delta argument")
        return

    # Iterate over the logs and print the ones that are within the specified time delta
    for log in logs:
        log_time = datetime.strptime(log[:19], '%Y-%m-%d %H:%M:%S')
        time_difference = (current_time - log_time).total_seconds()
        if time_difference <= time_delta:
            print(log)

def setup_migration_logger():
    logger = logging.getLogger("history")
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler('logs/history.log')
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    logger.addHandler(handler)

def main():
    args = parse_args(sys.argv[1:])

    if args.command == 'run':
        run(args)
    elif args.command == 'history':
        print_history_logs(args.duration)
    else:
        print('Invalid command')
        sys.exit(1)

if __name__ == '__main__':
    main()