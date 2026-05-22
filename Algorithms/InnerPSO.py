#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/8/20
# @Author  : github.com/guofei9987 - Modified by Nastaran Kianersi

import random

import matplotlib.pyplot as plt
import numpy as np
import subprocess
import time
import os
import copy

from scipy.special import expit  # numerically stable sigmoid

import csv
import re
from pathlib import Path
from datetime import datetime
import traceback

from . import utils
from .DecisionVar import *

import re
import csv

HEADER_RE = re.compile(r'^\s*log\s*,\s*score\s*$', re.I)

def parse_revision_score_table(lines_iter):
    """
    Parse a single revision score table from an iterator of lines.
    Returns (total, count) for the scores found, or (0, 0) if no valid table.
    """
    total = 0.0
    count = 0
    header_found = False
    
    # Find header
    for line in lines_iter:
        if HEADER_RE.search(line):
            header_found = True
            break
    
    if not header_found:
        return 0.0, 0
    
    # Parse data rows
    for line in lines_iter:
        # Skip empty lines
        if not line.strip():
            continue
            
        # End of table: lines starting with < (like <average>, <stddev>) EXCEPT for <global> which is part of the table
        if line.strip().startswith('<'):  
            parts = line.split(',')
            if parts[0].strip() != '<global>':
                break
        
        # Check if line contains comma
        if ',' not in line:
            break
        
        row = next(csv.reader([line]))
        if len(row) < 2:
            break
        
        # Score is in the second column (index 1)
        cell = row[1].strip()
        try:
            val = float(cell)
            total += val
            count += 1
        except ValueError:
            # Try cleaning the cell value
            try:
                val = float(re.sub(r'[^0-9eE+\-\.]', '', cell))
                total += val
                count += 1
            except ValueError:
                continue
    
    return total, count

def avg_revision_scores_multi(path):
    """
    Parse revision scores from a file that may contain one or two score tables.
    Returns a list of (average, count) tuples, one per table found.
    """
    results = []
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        # Look for "Starting validate-revision-score" marker
        if 'validate-revision-score' in lines[i].lower():
            # Parse the table starting from the next line
            total, count = parse_revision_score_table(iter(lines[i+1:]))
            if count > 0:
                results.append((total / count, count))
            else:
                results.append((0.0, 0))
            i += 1 # more efficient if i += count + 4
        else:
            i += 1
    
    return results if results else [(0.0, 0), (0.0, 0)]


class InnerPSO():
    r"""
    Binary PSO (Particle swarm optimization) algorithm.

    This code is derived from scikit-opt library and tailored specifically for optimizating
    the heat pump mining process.
    see https://scikit-opt.github.io/scikit-opt/#/en/README?id=_3-psoparticle-swarm-optimization
    
    This algorithm was adapted from the earlier works of J. Kennedy and
    R.C. Eberhart in Particle Swarm Optimization [IJCNN1995]_.
    """
    def __init__(self, event_sel_args, key_prop, pop, max_iter,
                 w=0.8, c1=1.49, c2=1.49, auto_coeffs=True, skip_event_sel_opt=False,
                 skip_learner_opt=False, early_termination=False, verbose=False):

        self.initially_called = True
        self.event_sel_args = event_sel_args
        self.key_prop = key_prop
        index = self.event_sel_args.index(self.key_prop)      # move the key property to the end of the list
        self.event_sel_args.append(self.event_sel_args.pop(index))
        self.event_sel_mapping = self.event_sel_args[:-1] #self.event_sel_args[:-1]  # the mapping for the binary decision variable in the event selection step (exclude the key property)
        
        self.skip_event_sel_opt = skip_event_sel_opt
        self.skip_learner_opt = skip_learner_opt
        self.default_learner_decision_var = LearnerDecisionVar(0.05, 1, 0.05, 1, 0)
        self.default_learner_args = [str(arg) for _, arg in enumerate(self.default_learner_decision_var.to_array(dtype=object))]
        
        self.pop = pop                                      # number of particles
        self.n_dim_event_sel = len(event_sel_args) - 1      # number of variables to choose from in the event selection step (excluding the key property)
        self.n_dim_learner   = LearnerDecisionVar.n_dim     # number of variables to optimize in the learner step
        self.X = np.array([InnerLoopDecisionVar.random(self.n_dim_event_sel, self.event_sel_mapping) for _ in range(pop)])
        if skip_event_sel_opt:
            for i in range(pop):
                self.X[i].bin_decision_var = BinaryDecisionVar.from_array(np.array([1]*self.n_dim_event_sel), self.event_sel_mapping)
        if skip_learner_opt:
            for i in range(pop):
                self.X[i].learner_decision_var = self.default_learner_decision_var
        
        self.iter_num = 0
        self.init_w = w
        self.init_cp, self.init_cg = c1, c2
        
        self.w = w                  # inertia
        self.cp, self.cg = c1, c2   # parameters to control personal best, global best respectively
        self.max_iter = max_iter 
        self.auto_coeffs = auto_coeffs
        self.early_termination = early_termination  # whether to terminate early if converged
        self.verbose = verbose                      # print the result of each iter or not

        self.v_max = 4
        self.V_event_sel = np.random.uniform(low=-self.v_max, high=self.v_max, size=(self.pop, self.n_dim_event_sel))  # speed of particles
        
        self.lb, self.ub = LearnerDecisionVar.lb, LearnerDecisionVar.ub
        assert self.n_dim_learner == len(self.lb) == len(self.ub), 'dim == len(lb) == len(ub) is not True'
        assert np.all(self.ub > self.lb), 'upper-bound must be greater than lower-bound'
        self.v_high = self.ub - self.lb
        self.V_learner = np.random.uniform(low=-self.v_high, high=self.v_high, size=(self.pop, self.n_dim_learner))
        self.prev_out_of_bound_learner = [np.array([False] * self.n_dim_learner) for _ in range(self.pop)]  # for handling boundary conditions in learner decision variables
        
        self.Y = np.array([0] * pop)
        self.pbest_x = np.array([copy.deepcopy(self.X[i]) for i in range(self.pop)])
        self.pbest_y = np.array([np.inf] * pop)         # best image of every particle in history
        self.gbest_x = InnerLoopDecisionVar.from_array(np.round(np.mean(np.array([self.pbest_x[i].bin_decision_var.arr.copy() for i in range(pop)]), axis=0)),
                                                                np.mean(np.array([self.pbest_x[i].learner_decision_var.to_array() for i in range(pop)]), axis=0),
                                                                self.event_sel_mapping)  # global best location among all particles
        self.gbest_y = np.inf                           # global best y for all particles
        self.gbest_y_hist = []                          # gbest_y of every iteration
        
        self.update_gbest()
        self.update_coeffs()
        
        # record verbose values
        self.record_mode = False
        self.record_value = {'X': [], 'V': [], 'Y': []}
        self.best_x, self.best_y = self.gbest_x, self.gbest_y  # history reasons, will be deprecated
        
    def reset_iter_params(self, gbest_x_init=None):
        if self.initially_called:
            # To make the random variables different across different runs AND the concurrent runs 
            # in different processes
            # If reproducibility is desired, change the seed to a constant
            np.random.seed(os.getpid()) 
            self.initially_called = False
        
        self.iter_num = 0
        self.w = self.init_w
        self.cp, self.cg = self.init_cp, self.init_cg
        
        if gbest_x_init is None:
            # print("gbest_x_init is None, randomly initializing particles")
            self.X = np.array([InnerLoopDecisionVar.random(self.n_dim_event_sel, self.event_sel_mapping) for _ in range(self.pop)])
            if self.skip_event_sel_opt:
                for i in range(self.pop):
                    self.X[i].bin_decision_var = BinaryDecisionVar.from_array(np.array([1]*self.n_dim_event_sel), self.event_sel_mapping)
            if self.skip_learner_opt:
                for i in range(self.pop):
                    self.X[i].learner_decision_var = self.default_learner_decision_var
            
            self.pbest_x = np.array([copy.deepcopy(self.X[i]) for i in range(self.pop)])
            self.gbest_x = InnerLoopDecisionVar.from_array(np.round(np.mean(np.array([self.pbest_x[i].bin_decision_var.arr.copy() for i in range(self.pop)]), axis=0)),
                                                           np.mean(np.array([self.pbest_x[i].learner_decision_var.to_array() for i in range(self.pop)]), axis=0),
                                                           self.event_sel_mapping)

        else:
            # print("gbest_x_init is provided, initializing particles around gbest_x_init: ", gbest_x_init)
            self.gbest_x = copy.deepcopy(gbest_x_init)
            self.X = np.array([InnerLoopDecisionVar.from_array(self.gbest_x.bin_decision_var.arr.copy(),
                                                               self.gbest_x.learner_decision_var.to_array(),
                                                               self.event_sel_mapping) for _ in range(self.pop)])
            self.pbest_x = np.array([copy.deepcopy(self.X[i]) for i in range(self.pop)])
            
        self.V_event_sel = np.random.uniform(low=-self.v_max, high=self.v_max, size=(self.pop, self.n_dim_event_sel))  # speed of particles
        self.V_learner = np.random.uniform(low=-self.v_high, high=self.v_high, size=(self.pop, self.n_dim_learner))
        
        self.Y = np.array([0] * self.pop)
        self.pbest_y = np.array([np.inf] * self.pop)         # best image of every particle in history
        self.gbest_y = np.inf                           # global best y for all particles
        self.gbest_y_hist = []                          # gbest_y of every iteration
        
        self.update_gbest()
        self.update_coeffs()
        
        # record verbose values
        self.record_mode = False
        self.record_value = {'X': [], 'V': [], 'Y': []}
        self.best_x, self.best_y = self.gbest_x, self.gbest_y  # history reasons, will be deprecated

    def update_V(self):
        if not self.skip_event_sel_opt:
            self.update_V_event_sel()
        if not self.skip_learner_opt:
            self.update_V_learner()
             
    def update_V_event_sel(self):
        r1 = np.random.rand(self.pop, self.n_dim_event_sel)
        r2 = np.random.rand(self.pop, self.n_dim_event_sel)
        intrm1 = np.array([self.pbest_x[i].bin_decision_var.arr  - self.X[i].bin_decision_var.arr for i in range(self.pop)])
        intrm2 = np.array([self.gbest_x.bin_decision_var.arr  - self.X[i].bin_decision_var.arr for i in range(self.pop)])
        
        self.V_event_sel = self.w * self.V_event_sel + \
                 self.cp * r1 * intrm1 + \
                 self.cg * r2 * intrm2
                 
        # Velocity clamping: ensure all velocities are at most v_max (no lower bound)
        np.minimum(self.V_event_sel, self.v_max, out=self.V_event_sel)
               
    def update_V_learner(self):
        # Set velocity to zero for dimensions that were previously out of bounds
        for i in range(self.pop):
            self.V_learner[i][self.prev_out_of_bound_learner[i]] = 0
            
        r1 = np.random.rand(self.pop, self.n_dim_learner)
        r2 = np.random.rand(self.pop, self.n_dim_learner)
        intrm1 = np.array([self.pbest_x[i].learner_decision_var.to_array() - self.X[i].learner_decision_var.to_array() for i in range(self.pop)])
        intrm2 = np.array([self.gbest_x.learner_decision_var.to_array() - self.X[i].learner_decision_var.to_array() for i in range(self.pop)])
        
        self.V_learner = self.w * self.V_learner + \
                 self.cp * r1 * intrm1 + \
                 self.cg * r2 * intrm2
     
    def update_X(self):
        if not self.skip_event_sel_opt:
            self.update_X_event_sel()
        if not self.skip_learner_opt:
            self.update_X_learner()
    
    def update_X_event_sel(self):
        sigmoid_V = expit(self.V_event_sel)
        r = np.random.random_sample(size=(self.pop, self.n_dim_event_sel))
        tmp_X = np.array([copy.copy(self.X[i].bin_decision_var.arr) for i in range(self.pop)])
        tmp_X = np.where(r < sigmoid_V, 1, 0)
        
        # If a particle has all zeros, randomly set one element to 1 => check if there's a better way
        for i in range(self.pop):
            if np.all(tmp_X[i] == 0):
                tmp_X[i][np.random.randint(self.n_dim_event_sel)] = 1
                
        for i in range(self.pop):
            self.X[i].bin_decision_var = BinaryDecisionVar.from_array(tmp_X[i], mapping=self.event_sel_mapping)
        
    def update_X_learner(self):
        for i in range(self.pop):
            self.X[i].learner_decision_var = LearnerDecisionVar.from_array(self.X[i].learner_decision_var.to_array() + self.V_learner[i])
            self.prev_out_of_bound_learner[i] = self.X[i].learner_decision_var.handle_boundary_pBest_dim(self.pbest_x[i].learner_decision_var.to_array())
            
    def update_coeffs(self):
        if self.auto_coeffs:
            t = self.iter_num
            n = self.max_iter
            self.w = (0.4/n**2) * (t - n) ** 2 + 0.4
            self.cp = -3 * t / n + 3.5
            self.cg =  3 * t / n + 0.5
            
    # Call a single job for a parent particle and get the result
    # Used when we wanna bypass the inner loop and use all variables in the event selection step
    # AND use some default values in the learning step
    def run_inner_loop_once(self, save_dir, event_trace_filename, parent_particle_idx):
        job_out_file_dir = str(os.path.join(save_dir, f'learner_job', ''))
        os.makedirs(job_out_file_dir, exist_ok=True)
        
        event_sel_args = self.event_sel_args
        learner_args = self.default_learner_args
        
        cmd = ["bash", "./submit.sh",  job_out_file_dir, './learnerv4_job_base.sh', f'./runtime_scripts/learner_job_{parent_particle_idx}.sh']
        subprocess.run(cmd)
        
        cmd = ["sbatch", "--parsable", f"./runtime_scripts/learner_job_{parent_particle_idx}.sh", str(event_trace_filename)] + learner_args + event_sel_args 
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) 
        
        job_id = proc.stdout.strip()
        print(f"InnerJob {job_id} args: {event_sel_args}")
        
        while True:
            if utils.is_job_completed_sacct(job_id):     
                out_file = os.path.join(job_out_file_dir, f"job_{job_id}.out")
                resutls = avg_revision_scores_multi(out_file)
                
                if len(resutls) == 1:
                    avg, n = resutls[0]
                    print(f"Job {job_id}, Parent prt {parent_particle_idx} -- Train: Rows: {n}, avg: {-avg}")
                else:
                    train_avg, train_n = resutls[0]  # use the first table (train data)
                    avg, n = resutls[1]  # use the second table (test data)
                    print(f"Job {job_id}, Parent prt {parent_particle_idx} -- Train: Rows {train_n}, avg {-train_avg} -- Test: Rows {n}, avg {-avg}")
                    
                result = -avg # Minimize negative score (== maximize score)
                break
                
            time.sleep(2)
             
        self.gbest_x = InnerLoopDecisionVar(BinaryDecisionVar.from_array(np.array([1]*self.n_dim_event_sel), mapping=self.event_sel_mapping),
                                            learner_decision_var=self.default_learner_decision_var)  # all variables used
        self.gbest_y = result
        self.best_x, self.best_y = self.gbest_x, self.gbest_y
        return self
        
    # Use subprocess to run the job script and get the results
    def call_scripts(self, save_dir, event_trace_filename, parent_particle_idx):        
        job_out_file_dir = str(os.path.join(save_dir, f'inner_loop_iter_{self.iter_num}', ''))
        os.makedirs(job_out_file_dir, exist_ok=True)
        job_ids = []
        results = [0] * self.pop
        
        for idx, x in enumerate(self.X):
            event_sel_args = [self.event_sel_args[i] for i, val in enumerate(x.bin_decision_var.arr) if val == 1] + [self.event_sel_args[-1]] if not self.skip_event_sel_opt else self.event_sel_args
            # event_sel_args = [self.event_sel_args[i] for i, val in enumerate(x.bin_decision_var.arr) if val == 1] if not self.skip_event_sel_opt else self.event_sel_args  #+ [self.event_sel_args[-1]]
            learner_args = [str(arg) for _, arg in enumerate(x.learner_decision_var.to_array(dtype=object))] if not self.skip_learner_opt else self.default_learner_args
            
            cmd = ["bash", "./submit.sh",  job_out_file_dir, './learnerv4_job_base.sh', f'./runtime_scripts/learner_job_{parent_particle_idx}.sh']
            subprocess.run(cmd)
            
            cmd = ["sbatch", "--parsable", f"./runtime_scripts/learner_job_{parent_particle_idx}.sh", str(event_trace_filename)] + learner_args + event_sel_args 
            time.sleep(np.random.randint(2) + 1)
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) 
            
            # stdout = proc.stdout.readline()
            # job_id = stdout.strip().split()[-1] if stdout.strip().startswith("Submitted batch job") else None
            job_id = proc.stdout.strip()
            print(f"InnerJob {job_id} args: {event_sel_args} and {learner_args}")
            job_ids.append(int(job_id))
            
        # Check for finished jobs by checking their output files
        # Collect the results from each output file when it finished
        pending = list(range(self.pop))
        while pending:
            for idx in pending:
                if utils.is_job_completed_sacct(job_ids[idx]):     
                    out_file = os.path.join(job_out_file_dir, f"job_{job_ids[idx]}.out")
                    resutls = avg_revision_scores_multi(out_file)
                    
                    if len(resutls) == 1:
                        avg, n = resutls[0]
                        print(f"Job {job_ids[idx]}, Prtc {idx} -- Train: Rows {n}, avg {-avg}")
                    else:
                        train_avg, train_n = resutls[0]  # use the first table (train data)
                        avg, n = resutls[1]  # use the second table (test data)
                        print(f"Job {job_ids[idx]}, Prtc {idx} -- Train: Rows {train_n}, avg {-train_avg} -- Test: Rows {n}, avg {-avg}")
                        
                    pending.remove(idx)
                    results[idx] = -avg # Minimize negative score (== maximize score)
                
            time.sleep(2)
            
        self.Y = np.array(results)  
        return self.Y
            
    def update_pbest(self):
        '''
        personal best
        :return:
        '''
        self.need_update = self.pbest_y > self.Y
        # print("need_update: ", self.need_update)
        for i in range(self.pop):
            if self.need_update[i]:
                # print(f"Updating pbest for particle {i} from {self.pbest_x[i]} to {self.X[i]}")
                self.pbest_x[i] = copy.deepcopy(self.X[i])
                # print("After update, pbest_x: ", self.pbest_x[i])
        # print("self.Y: ", self.Y, ", pbest_y before update: ", self.pbest_y)
        self.pbest_y = np.where(self.need_update, self.Y, self.pbest_y)

    def update_gbest(self):
        '''
        global best
        :return:
        '''
        idx_min = self.pbest_y.argmin()
        # print("idx_min: ", idx_min, "self.pbest_y: ", self.pbest_y)
        # print("pbest_y[idx_min]: ", self.pbest_y[idx_min], "current gbest_y: ", self.gbest_y)
        # #print("InnerPSO update_gbest: ", self.gbest_y, self.pbest_y.shape, self.pbest_y[idx_min])
        if self.gbest_y > self.pbest_y[idx_min]:
            # print("Updating gbest_x: from ", self.gbest_x, " to ", self.X[idx_min])
            # print("AND Updating gbest: from ", self.gbest_y, " to ", self.pbest_y[idx_min])
            self.gbest_x = copy.deepcopy(self.X[idx_min])
            self.gbest_y = self.pbest_y[idx_min]
        # else:
        #     print("NOT Updating gbest")

    def recorder(self):
        if not self.record_mode:
            return
        self.record_value['X'].append(self.X)
        self.record_value['V'].append(self.V)
        self.record_value['Y'].append(self.Y)
    
    def iterate(self, iter_num, save_dir, event_trace_filename, parent_particle_idx):
        converged = False
        self.iter_num = iter_num
        
        self.update_V()
        self.recorder()
        self.update_X()
        self.call_scripts(save_dir, event_trace_filename, parent_particle_idx) # Call the jobs and wait for them to finish, then the returned value would be the value
                                                                               # of y for each particle. Run a separate job for each particle.
        self.update_pbest()
        self.update_gbest()
        self.update_coeffs()
        
        # if self.verbose:
            #print('Iter: {}, Best fit: {} at {}'.format(iter_num, self.gbest_y, self.gbest_x))

        self.gbest_y_hist.append(self.gbest_y)
        
        if self.early_termination:
            if self.analyze_convergence():
                print(f"Converged at iteration {iter_num}. Stopping early.")
                converged = True
            
        return converged
    
    def optimize_distributed(self, save_dir, event_trace_filename, parent_particle_idx, gbest_x_init=None):
        self.reset_iter_params(gbest_x_init)
        # print(f"InnerPSO: max_iter: {self.max_iter}, pop: {self.pop}")
        
        for iter_num in range(self.max_iter):
            print(f"InnerPSO iteration: {iter_num}")
            converged = self.iterate(iter_num, save_dir, event_trace_filename, parent_particle_idx)
            if converged:
                break
            
        self.best_x, self.best_y = self.gbest_x, self.gbest_y
        # print("best_x: ", self.best_x, "best_y: ", self.best_y)
        return self
    
    def analyze_convergence(self):
        try:
            gbest_converged = self.has_converged_gbest(patience=min(5, self.max_iter//4))
            return gbest_converged
        
        except Exception as e:
            traceback.print_exc()
            print(f"Error in convergence analyzer: {e}")
        
    def has_converged_gbest(self, patience=10, rel_tol=1e-4):
        """
        Check convergence based on stability of global best fitness.
        Args:
            patience (int): Number of iterations to look back for improvement.
            rel_tol (float): Relative tolerance to detect "no significant improvement".
                            Automatically scaled relative to fitness values.
        Returns:
            bool: True if gbest has converged, False otherwise.
        """
        if len(self.gbest_y_hist) < patience + 1:
            return False

        if abs(self.gbest_y) < 1e-12:
            return False
        
        recent = self.gbest_y_hist[-patience:]
        past = self.gbest_y_hist[-(patience+1)]
        
        improvement = abs(recent[-1] - past)
        scale = max(abs(past), 1.0)  # avoid division by zero
        rel_improvement = improvement / scale
        
        status = "converged" if rel_improvement < rel_tol else "not converged"
        print(f"InnerPSO gbest improvement analysis: {status}")
        return rel_improvement < rel_tol

    # TODO: draw a table containing the value of each dimension for each particle over iterations
    def plot_particle_evolution(self):
        """
        For each particle, plot the evolution of each..
        """
        try:
            pass
        except Exception as e:
            traceback.print_exc()
            print(f"Error plotting particle evolution: {e}")
    
    def plot_gbest_y_hist(self):
        """
        Plot gbest_y_hist which is the best y value among all particles in each iteration.
        """
        try:
            pass
        except Exception as e:
            traceback.print_exc()
            print(f"Error plotting evolution: {e}")
        
    def visualize(self):
        self.plot_particle_evolution()
        self.plot_gbest_y_hist()