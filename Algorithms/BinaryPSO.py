#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/8/20
# @Author  : github.com/guofei9987 - Modified by Nastaran Kianersi

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


HEADER_RE = re.compile(r'^\s*LOG-?NAME\s*,\s*REVISION[-_ ]?SCORE(S)?\s*,', re.I)
REV_COL_RE = re.compile(r'^REVISION[-_ ]?SCORE(S)?$', re.I)

def avg_revision_scores(path):
    total = 0.0
    count = 0
    in_table = False
    rev_idx = None

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not in_table:
                if HEADER_RE.search(line):
                    # parse header to find the revision score column index
                    header = next(csv.reader([line]))
                    header = [h.strip() for h in header]
                    for i, h in enumerate(header):
                        if REV_COL_RE.match(h):
                            rev_idx = i
                            break
                    if rev_idx is None:
                        raise RuntimeError("Could not find REVISION-SCORE(S) column in header")
                    in_table = True
                # else: keep skipping lines until we hit the header
                continue

            # end of table heuristics: stop if we hit a blank line or a non-CSV-ish line
            if not line.strip() or ',' not in line:
                break

            row = next(csv.reader([line]))
            if len(row) <= rev_idx:
                break

            cell = row[rev_idx].strip()
            # strip any stray characters (e.g., percentages) and convert to float
            try:
                val = float(cell)
            except ValueError:
                val = float(re.sub(r'[^0-9eE+\-\.]', '', cell))

            total += val
            count += 1

    if count == 0:
        return 0, 0
    return total/count, count


class BinaryPSO():
    """
    Binary PSO (Particle swarm optimization) algorithm.

    This code is derived from scikit-opt library and tailored specifically for optimizating
    the heat pump mining process.
    see https://scikit-opt.github.io/scikit-opt/#/en/README?id=_3-psoparticle-swarm-optimization
    
    This algorithm was adapted from the earlier works of J. Kennedy and
    R.C. Eberhart in Particle Swarm Optimization [IJCNN1995]_.

    The position update can be defined as:

    .. math::
       x_{i}(t+1) = x_{i}(t) + v_{i}(t+1)

    Where the position at the current step :math:`t` is updated using
    the computed velocity at :math:`t+1`. Furthermore, the velocity update
    is defined as:

    .. math::

       v_{ij}(t + 1) = w * v_{ij}(t) + c_{p}r_{1j}(t)[y_{ij}(t) − x_{ij}(t)]
                       + c_{g}r_{2j}(t)[\hat{y}_{j}(t) − x_{ij}(t)]

    Here, :math:`cp` and :math:`cg` are the cognitive and social parameters
    respectively. They control the particle's behavior given two choices: (1) to
    follow its *personal best* or (2) follow the swarm's *global best* position.
    Overall, this dictates if the swarm is explorative or exploitative in nature.
    In addition, a parameter :math:`w` controls the inertia of the swarm's
    movement.

    """
    def __init__(self, learner_args, key_prop, pop, max_iter,
                 w=0.8, c1=1, c2=1, auto_coeffs=True, early_termination=False, verbose=False):

        self.learner_args = learner_args
        self.key_prop = key_prop
        index = self.learner_args.index(self.key_prop)   # move the key property to the end of the list
        self.learner_args.append(self.learner_args.pop(index))
        
        self.pop = pop  # number of particles
        self.n_dim = len(learner_args)  - 1   # dimension of decision variable
        self.X = np.array([BinaryDecisionVar.random(self.n_dim) for _ in range(pop)])    #np.random.choice([0, 1], size=(self.pop, self.n_dim))
        self.iter_num = 0
        self.init_w = w
        self.init_cp, self.init_cg = c1, c2
        
        self.w = w  # inertia
        self.cp, self.cg = c1, c2  # parameters to control personal best, global best respectively
        self.max_iter = max_iter 
        self.auto_coeffs = auto_coeffs
        self.early_termination = early_termination  # whether to terminate early if converged
        self.verbose = verbose  # print the result of each iter or not

        self.v_max = 4
        self.V = np.random.uniform(low=-self.v_max, high=self.v_max, size=(self.pop, self.n_dim))  # speed of particles
        self.Y = np.array([0] * pop)
        self.pbest_x = self.X.copy()                    # personal best location of every particle in history
        self.pbest_y = np.array([np.inf] * pop)         # best image of every particle in history
        # self.pbest_x[np.random.randint(self.pop)]
        self.gbest_x = BinaryDecisionVar.from_array(np.round(np.mean(np.array([self.pbest_x[i].arr for i in range(pop)]), axis=0)))
        self.gbest_y = np.inf                           # global best y for all particles
        self.gbest_y_hist = []                          # gbest_y of every iteration
        
        self.update_gbest()
        self.update_coeffs()
        
        # record verbose values
        self.record_mode = False
        self.record_value = {'X': [], 'V': [], 'Y': []}
        self.best_x, self.best_y = self.gbest_x, self.gbest_y  # history reasons, will be deprecated
        
    def reset_iter_params(self):
        self.iter_num = 0
        self.w = self.init_w
        self.cp, self.cg = self.init_cp, self.init_cg
        self.update_coeffs()
        
    def update_V(self):
        r1 = np.random.rand(self.pop, self.n_dim)
        r2 = np.random.rand(self.pop, self.n_dim)
        intrm1 = np.array([self.pbest_x[i].arr  - self.X[i].arr for i in range(self.pop)])
        intrm2 = np.array([self.gbest_x.arr  - self.X[i].arr for i in range(self.pop)])
        
        self.V = self.w * self.V + \
                 self.cp * r1 * intrm1 + \
                 self.cg * r2 * intrm2
                 
        # Velocity clamping: ensure all velocities are at most v_max (no lower bound)
        np.minimum(self.V, self.v_max, out=self.V)
                
    def update_X(self):
        sigmoid_V = expit(self.V)
        r = np.random.random_sample(size=(self.pop, self.n_dim))
        tmp_X = np.array([copy.copy(self.X[i].arr) for i in range(self.pop)])
        tmp_X = np.where(r < sigmoid_V, 1, 0)
        self.X = np.array([BinaryDecisionVar.from_array(tmp_X[i]) for i in range(self.pop)])
        
    def update_coeffs(self):
        if self.auto_coeffs:
            t = self.iter_num
            n = self.max_iter
            self.w = (0.4/n**2) * (t - n) ** 2 + 0.4
            self.cp = -3 * t / n + 3.5
            self.cg =  3 * t / n + 0.5
            
    # Use subprocess to run the job script and get the results
    def call_scripts(self, save_dir, event_trace_filename, parent_particle_idx):        
        job_out_file_dir = str(os.path.join(save_dir, f'inner_loop_iter_{self.iter_num}', ''))
        os.makedirs(job_out_file_dir, exist_ok=True)
        job_ids = []
        results = [0] * self.pop
        
        for idx, x in enumerate(self.X):
            x = copy.copy(x)
            args = [self.learner_args[i] for i, val in enumerate(x.arr) if val == 1] + [self.learner_args[-1]]  
            cmd = ["bash", "./submit.sh",  job_out_file_dir, './learner_job_base.sh', f'./learner_job_{parent_particle_idx}.sh']
            subprocess.run(cmd)
            
            cmd = ["sbatch", "--parsable", f"./learner_job_{parent_particle_idx}.sh", str(event_trace_filename)] + args 
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) 
           
            # stdout = proc.stdout.readline()
            # job_id = stdout.strip().split()[-1] if stdout.strip().startswith("Submitted batch job") else None
            job_id = proc.stdout.strip()
            print("Submitting inner job", job_id, "with args:", args)
            job_ids.append(int(job_id))
            
        # Check for finished jobs by checking their output files
        # Collect the results from each output file when it finished
        pending = list(range(self.pop))
        while pending:
            for idx in pending:
                # state = subprocess.run(
                #                 ["sacct", "-j", str(job_ids[idx]), "--noheader", "--format=State"],
                #                 capture_output=True, text=True
                #             )
                # print(state)
                # .stdout.strip().splitlines()[-1].strip()
                state = utils.get_job_state(job_ids[idx])
                
                if state == 'COMPLETED':        
                    out_file = os.path.join(job_out_file_dir, f"job_{job_ids[idx]}.out")
                    avg, n = avg_revision_scores(out_file)
                    print(f"Job ID {job_ids[idx]}, Particle {idx} --- Rows: {n}, Avg(REVISION-SCORE[S]): {-avg}")
                    pending.remove(idx)
                    results[idx] = -avg # Minimize negative score (== maximize score)
                
            time.sleep(5)
            
        self.Y = np.array(results)  
        return self.Y
            
    def update_pbest(self):
        '''
        personal best
        :return:
        '''
        self.need_update = self.pbest_y > self.Y
        # print("BinaryPSO update_pbest: ", self.need_update.shape, self.pbest_y.shape, self.Y.shape)
        # [:, np.newaxis]
        self.pbest_x = np.where(self.need_update, self.X, self.pbest_x)
        self.pbest_y = np.where(self.need_update, self.Y, self.pbest_y)

    def update_gbest(self):
        '''
        global best
        :return:
        '''
        idx_min = self.pbest_y.argmin()
        # print("BinaryPSO update_gbest: ", self.gbest_y, self.pbest_y.shape, self.pbest_y[idx_min])
        if self.gbest_y > self.pbest_y[idx_min]:
            self.gbest_x = self.X[idx_min]
            self.gbest_y = self.pbest_y[idx_min]

    def recorder(self):
        if not self.record_mode:
            return
        self.record_value['X'].append(self.X)
        self.record_value['V'].append(self.V)
        self.record_value['Y'].append(self.Y)
    
    def iterate(self, iter_num, save_dir, event_trace_filename, parent_particle_idx):
        converged = False
        
        self.iter_num = iter_num
        print("Iter", iter_num, "Coeffs: w =", self.w, ", cp =", self.cp, ", cg =", self.cg)
        self.update_V()
        self.recorder()
        self.update_X()
        self.call_scripts(save_dir, event_trace_filename, parent_particle_idx) # Call the jobs and wait for them to finish, then the returned value would be the value
                            # of y for each particle. Run a separate job for each particle.
        self.update_pbest()
        self.update_gbest()
        self.update_coeffs()
        
        if self.verbose:
            print('Iter: {}, Best fit: {} at {}'.format(iter_num, self.gbest_y, self.gbest_x))

        self.gbest_y_hist.append(self.gbest_y)
        
        if self.early_termination:
            if self.analyze_convergence():
                print(f"Converged at iteration {iter_num}. Stopping early.")
                converged = True
            
        return converged
    
    def optimize_distributed(self, save_dir, event_trace_filename, parent_particle_idx):
        self.reset_iter_params()
        print("BinaryPSO: max_iter:", self.max_iter, "pop:", self.pop)
        
        for iter_num in range(self.max_iter):
            print("BinaryPSO iteration:", iter_num)
            converged = self.iterate(iter_num, save_dir, event_trace_filename, parent_particle_idx)
            if converged:
                break
            
        self.best_x, self.best_y = self.gbest_x, self.gbest_y
        return self #self.best_x, self.best_y
    
    def analyze_convergence(self):
        try:
            gbest_converged = self.has_converged_gbest()
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

        recent = self.gbest_y_hist[-patience:]
        past = self.gbest_y_hist[-(patience+1)]
        
        improvement = abs(recent[-1] - past)
        scale = max(abs(past), 1.0)  # avoid division by zero
        rel_improvement = improvement / scale
        
        print("gbest improvement analysis:", "converged" if rel_improvement < rel_tol else "not converged")
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