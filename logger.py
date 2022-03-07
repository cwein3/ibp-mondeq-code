'''
Class for writing experimental logs. 
'''
import numpy as np 
import pandas as pd 
import os
import pickle
import os
import torch

from datetime import datetime

def create_metric(metric_type):
	if metric_type == 'avg':
		return AvgMetric()

	if metric_type == 'max':
		return MaxMetric()

	if metric_type == 'min': 
		return MinMetric()

class Metric(object):

	def __init__(self):
		self.curr_val = None

	def update(self):
		raise NotImplementedError

	def val(self):
		return self.curr_val

	def reset(self):
		self.curr_val = None
		
class MinMetric(Metric):

	def __init__(self):
		super().__init__()
		self.curr_val = np.inf


	def update(self, update_val):
		self.curr_val = min(self.curr_val, update_val)
		return self.curr_val

class MaxMetric(Metric):

	def __init__(self):
		super().__init__()
		self.curr_val = -np.inf

	def update(self, update_val):
		self.curr_val = max(self.curr_val, update_val)
		return self.curr_val

class AvgMetric(Metric):

	def __init__(self):
		super().__init__()
		self.curr_sum = 0
		self.tot_count = 0
		self.curr_val = 0 # average is 0 with 0 elements

	def update(self, update_val, count=1):
		self.tot_count += count
		self.curr_sum += update_val
		self.curr_val = self.curr_sum/self.tot_count
		return self.curr_val

	def reset(self):
		self.tot_count = 0
		self.curr_sum = 0
		super().reset()



def get_latest_run_id(dir_name):
	# assume dir names are run_0, run_1, run_2, etc. 
	curr_run = 0
	for subdir in os.listdir(dir_name):
		full_path = os.path.join(dir_name, subdir)
		if os.path.isdir(full_path):
			run_num = int(subdir.split('_')[-1])
			if run_num > curr_run:
				curr_run = run_num 
	return curr_run + 1

def get_run_dir(outer_dir, slurm_id, rank=None, run_id=None):
	if slurm_id is None:
		# then just make it the date
		slurm_id = datetime.utcnow().strftime("%H_%M_%S_%f-%d_%m_%y")

	slurm_dir = os.path.join(
		outer_dir,
		slurm_id)

	if rank is not None and torch.distributed.is_initialized():
		if is_main_process(rank):
			if not os.path.exists(slurm_dir):
				os.makedirs(slurm_dir)
		torch.distributed.barrier()
	else:
		if not os.path.exists(slurm_dir):
			os.makedirs(slurm_dir)

	if run_id is None:
		run_id = 'run_%d' % (get_latest_run_id(slurm_dir))
		
	return os.path.join(slurm_dir, run_id)

def is_main_process(rank):
	return rank == 0
	
class ExperimentLogWriter(object):

	def __init__(
		self,
		outer_dir,
		slurm_id, 
		rank=None,
		run_id=None):

		self.save_dir = get_run_dir(outer_dir, slurm_id, rank=rank, run_id=run_id)

		if not os.path.exists(self.save_dir) and (is_main_process(rank) or rank is None):
			os.makedirs(self.save_dir)

		self.log_loc = os.path.join(self.save_dir, 'log.txt')
		self.data_dict_dir = os.path.join(self.save_dir, 'data_dicts')
		if not os.path.exists(self.data_dict_dir) and (is_main_process(rank) or rank is None):
			os.makedirs(self.data_dict_dir)

		self.checkpoint_dir = os.path.join(self.save_dir, 'checkpoints')
		if not os.path.exists(self.checkpoint_dir) and (is_main_process(rank) or rank is None):
			os.makedirs(self.checkpoint_dir)

		self.rank = rank
		self.data_dicts = {} # data for storing
		self.metrics = {} 

	def save_args(
		self, 
		args):
		if self.rank is not None and not is_main_process(self.rank):
			return
		self.args_loc = os.path.join(self.save_dir, 'args.pkl')
		pickle.dump(args, open(self.args_loc, 'wb'))

		# also write the args to the log file
		args_dict = vars(args)
		with open(self.log_loc, 'a') as log_file:
			for arg in sorted (args_dict.keys()): 
				log_file.write(
					'{:20} : {}\n'.format(arg, args_dict[arg]))

	# logging
	def log(
		self,
		log_str):
		if self.rank is not None and not is_main_process(self.rank):
			return
		with open(self.log_loc, 'a') as log_file:
			log_file.write('{}\n'.format(log_str))

	# handle metric storing and updating
	def add_metric(
		self,
		metric_name, 
		metric_type='avg'):
		self.metrics[metric_name] = create_metric(metric_type)

	def get_metric(
		self,
		metric_name):
		return self.metrics[metric_name].val()

	def update_metric(
		self,
		metric_name,
		update_val,
		**update_kwargs):
		return self.metrics[metric_name].update(update_val, **update_kwargs)

	def reset_metric(
		self,
		metric_name):
		self.metrics[metric_name].reset()

	def reset_metrics(self):
		for metric in self.metrics.values():
			metric.reset()

	# handle data dictionary creation and saving
	def create_data_dict(
		self,
		col_names,
		dict_id='default'):
		if self.rank is not None and not is_main_process(self.rank):
			return
		df = pd.DataFrame({col_name : [] for col_name in col_names})
		self.data_dicts[dict_id] = df
		
	def update_data_dict(
		self,
		update_dict,
		dict_id='default'):
		if self.rank is not None and not is_main_process(self.rank):
			return
		df = pd.Series(update_dict)
		self.data_dicts[dict_id] = self.data_dicts[dict_id].append(df, ignore_index=True)
		
	def save_data_dict(
		self,
		dict_id='default'):
		if self.rank is not None and not is_main_process(self.rank):
			return
		dict_path = os.path.join(self.data_dict_dir, '{}.pkl'.format(dict_id))
		self.data_dicts[dict_id].to_pickle(dict_path)

	def save_data_dicts(
		self):
		if self.rank is not None and not is_main_process(self.rank):
			return
		for dict_id in self.data_dicts:
			self.save_data_dict(dict_id)

	# handle model checkpointing
	def ckpt_model(
		self,
		to_ckpt,
		index,
		is_latest = False):
		if self.rank is not None and not is_main_process(self.rank):
			return
		ckpt_str = 'latest_' if is_latest else ''
		ckpt_loc = os.path.join(self.checkpoint_dir, '{}{}.pth'.format(ckpt_str, index))

		# remove the previous if it is the latest
		if is_latest:
			for fname in os.listdir(self.checkpoint_dir):
				if 'latest_' in fname:
					os.remove(os.path.join(self.checkpoint_dir, fname))

		torch.save(to_ckpt, ckpt_loc)