import os
import pickle
import argparse

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres

from hpbandster.optimizers import BOHB

import logging
logging.basicConfig(level=logging.DEBUG)



parser = argparse.ArgumentParser(description='BOHB -- DeepFreak')
parser.add_argument('--min_budget',   type=float, help='Minimum number of epochs for training.',    default=1)
parser.add_argument('--max_budget',   type=float, help='Maximum number of epochs for training.',    default=50)
parser.add_argument('--n_iterations', type=int,   help='Number of iterations performed by the optimizer', default=16)
parser.add_argument('--worker', help='Flag to turn this into a worker process', action='store_true')
parser.add_argument('--run_id', type=str, help='A unique run id for this optimization run. An easy option is to use the job id of the clusters scheduler.', default="deepfreak")
parser.add_argument('--nic_name',type=str, help='Which network interface to use for communication.', default='lo')
parser.add_argument('--shared_directory',type=str, help='A directory that is accessible for all processes, e.g. a NFS share.', default='./results')
parser.add_argument('--previous_run_dir',type=str, help='A directory that contains a config.json and results.json for the same configuration space.', default=None)
parser.add_argument('--worker_id',type=str, help='An ID for the worker.', default=0)
parser.add_argument('--port', type=int, action='store', default=9090)
parser.add_argument('--train_path', dest='train_path', action='store', default='../data/synthetic/train/')
parser.add_argument('--val_path', dest='val_path', action='store', default='../data/real_preprocessed/validation/')
parser.add_argument('--name', dest='name', action='store', default='models')

args=parser.parse_args()

from deepfreak_worker import PyTorchWorker as worker

host = hpns.nic_name_to_host(args.nic_name)
port = args.port

# Start a worker instead of dispatcher
if args.worker:
  import time
  time.sleep(5)	# short artificial delay to make sure the nameserver is already running
  w = worker(run_id=args.run_id, host=host, timeout=120, id=args.worker_id, gpu_id=args.worker_id, train_path=args.train_path, val_path=args.val_path, name=args.name)
  w.load_nameserver_credentials(working_directory=args.shared_directory)
  w.run(background=False)
  exit(0)


# create a Result object.
result_logger = hpres.json_result_logger(directory=args.shared_directory, overwrite=True)


# Start a nameserver:
NS = hpns.NameServer(run_id=args.run_id, host=host, port=port, working_directory=args.shared_directory)
ns_host, ns_port = NS.start()

# Restore previous run
previous_run = None
if(args.previous_run_dir is not None):
  previous_run = hpres.logged_results_to_HBS_result(args.previous_run_dir)

# Run an optimizer
bohb = BOHB(
    configspace = worker.get_configspace(),
    run_id = args.run_id,
    host=host,
    nameserver=ns_host,
    nameserver_port=ns_port,
    result_logger=result_logger,
    min_budget=args.min_budget,
    max_budget=args.max_budget,
    previous_result=previous_run
    )
res = bohb.run(n_iterations=args.n_iterations, min_n_workers=1)

# store results
with open(os.path.join(args.shared_directory, 'results.pkl'), 'wb') as fh:
  pickle.dump(res, fh)

# shutdown
bohb.shutdown(shutdown_workers=True)
NS.shutdown()

