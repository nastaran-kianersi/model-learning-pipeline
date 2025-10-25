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
    def __init__(self, event_trace_addr, n_dim, pop, max_iter,
                 w=0.8, c1=1, c2=1, auto_coeffs=True, early_termination=False, verbose=False):

        self.event_trace_addr = event_trace_addr
        self.iter_num = 0
        self.n_dim = n_dim
        self.X = np.random.choice([0, 1], size=(self.pop, self.n_dim))
        
        self.w = w  # inertia
        self.cp, self.cg = c1, c2  # parameters to control personal best, global best respectively
        self.pop = pop  # number of particles
        self.max_iter = max_iter 
        self.auto_coeffs = auto_coeffs
        self.early_termination = early_termination  # whether to terminate early if converged
        self.verbose = verbose  # print the result of each iter or not

        self.v_max = 4
        self.V = np.random.uniform(low=-self.v_max, high=self.v_max, size=(self.pop, self.n_dim))  # speed of particles
        self.Y = np.array([0] * pop)
        self.pbest_x = self.X.copy()                    # personal best location of every particle in history
        self.pbest_y = np.array([np.inf] * pop)         # best image of every particle in history
        self.gbest_x = np.mean(self.pbest_x)    # global best location for all particles
        self.gbest_y = np.inf                           # global best y for all particles
        self.gbest_y_hist = []                          # gbest_y of every iteration
        
        self.update_gbest()
        self.update_coeffs()
        
        # record verbose values
        self.record_mode = False
        self.record_value = {'X': [], 'V': [], 'Y': []}
        self.best_x, self.best_y = self.gbest_x, self.gbest_y  # history reasons, will be deprecated
        

    def update_V(self):
        r1 = np.random.rand(self.pop, self.n_dim)
        r2 = np.random.rand(self.pop, self.n_dim)
        intrm1 = self.pbest_x - self.X
        intrm2 = self.gbest_x - self.X
        self.V = self.w * self.V + \
                 self.cp * r1 * intrm1 + \
                 self.cg * r2 * intrm2
                 
        # Velocity clamping: ensure all velocities are at most v_max (no lower bound)
        np.minimum(self.V, self.v_max, out=self.V)
                
    
    def update_X(self):
        sigmoid_V = expit(self.V)
        r = np.random.random_sample(self.pop)
        self.X = np.where(r > sigmoid_V, 1, 0)
    
    def update_coeffs(self):
        if self.auto_coeffs:
            t = self.iter_num
            n = self.max_iter
            self.w = (0.4/n**2) * (t - n) ** 2 + 0.4
            self.cp = -3 * t / n + 3.5
            self.cg =  3 * t / n + 0.5
            
    # Use subprocess to run the job script and get the results
    def call_scripts(self, save_dir, event_trace_addr, learner_args):        
        job_out_file_dir = str(os.path.join(save_dir, f'Iter_{self.iter_num}', ''))
        os.makedirs(job_out_file_dir, exist_ok=True)
        processes = []
        job_ids = []
        results = [0] * self.pop
        
        for idx, x in enumerate(self.X):
            x = copy.copy(x)
            args = [learner_args[i] for i, val in enumerate(x) if val == 1]            
            cmd = ["bash", "./submit.sh",  job_out_file_dir, './learner_job_base.sh', '/learner_job.sh']
            
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) # waits for the job to be submitted
            cmd = ["sbatch", "./learner_job.sh", str(event_trace_addr)] + args 
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) 
            processes.append(proc)
            
            stdout = proc.stdout.readline()
            job_id = stdout.strip().split()[-1] if stdout.strip().startswith("Submitted batch job") else None
            job_ids.append(int(job_id))
            print("Submitting job", job_id, "with args: ", ' '.join(args))
            
        # Check for finished jobs by checking their output files
        # Collect the results from each output file when it finished
        pending = list(range(self.pop))
        while pending:
            for idx in pending:
                if processes[idx].poll() is not None:  # Process finished
                    out_file = os.path.join(job_out_file_dir, f"job_{job_ids[idx]}.out")
                    avg, n = avg_revision_scores(out_file)
                    print(f"Job ID {job_ids[idx]}, Particle {idx} --- Rows: {n}, Avg(REVISION-SCORE[S]): {-avg}")
                    pending.remove(idx)
                    results[idx] = -avg # Minimize negative score (== maximize score)
                else:
                    time.sleep(5)
            
        self.Y = np.array(results)  
        return self.Y
            
    def update_pbest(self):
        '''
        personal best
        :return:
        '''
        self.need_update = self.pbest_y > self.Y
        self.pbest_x = np.where(self.need_update, self.X, self.pbest_x)
        self.pbest_y = np.where(self.need_update, self.Y, self.pbest_y)

    def update_gbest(self):
        '''
        global best
        :return:
        '''
        idx_min = self.pbest_y.argmin()
        if self.gbest_y > self.pbest_y[idx_min]:
            self.gbest_x = self.X[idx_min]
            self.gbest_y = self.pbest_y[idx_min]

    def recorder(self):
        if not self.record_mode:
            return
        self.record_value['X'].append(self.X)
        self.record_value['V'].append(self.V)
        self.record_value['Y'].append(self.Y)
    
    def iterate(self, iter_num, save_dir, event_trace_addr, learner_args):
        converged = False
        
        self.iter_num = iter_num
        print("Iter ", iter_num, "Coeffs: w=", self.w, " cp=", self.cp, " cg=", self.cg)
        self.update_V()
        self.recorder()
        self.update_X()
        self.call_scripts(save_dir, event_trace_addr, learner_args) # Call the jobs and wait for them to finish, then the returned value would be the value
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
    
    def optimize_distributed(self, save_dir, event_trace_addr, learner_args):
        for iter_num in range(self.max_iter):
            converged = self.iterate(iter_num, save_dir, event_trace_addr, learner_args)
            if converged:
                break
            
        self.best_x, self.best_y = self.gbest_x, self.gbest_y
        return self.best_x, self.best_y
    
    def analyze_convergence(self):
        try:
            gbest_converged = self.has_converged_gbest()
            pos_converged, vel_converged = self.has_converged_particles()
            return gbest_converged and (pos_converged or vel_converged)
        
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

    def has_converged_particles(self, tol_factor=1e-2):
        """
        Check convergence based on particle coalescence and velocity collapse.
        Args:
            tol_factor (float): Factor relative to position/velocity spread to determine convergence.
        (Plan to) Return:
            dict: {
                "position_converged": bool,
                "velocity_converged": bool
            }
        """
        positions = self.X
        velocities = self.V
        
        # Position convergence: low spread in particle positions
        pos_spread = np.mean(np.std(positions))
        pos_scale = np.mean(np.mean(np.abs(positions))) + 1e-9
        position_converged = pos_spread / pos_scale < tol_factor
        
        # Velocity convergence: near-zero average velocity
        vel_norms = np.linalg.norm(velocities, axis=1)
        velocity_converged = np.mean(vel_norms) < tol_factor * pos_scale
        
        print("Convergence analysis based on position:", "converged" if position_converged else "not converged")
        print("Convergence analysis based on velocity:", "converged" if velocity_converged else "not converged")
        
        return position_converged, velocity_converged
    
    def plot_particle_evolution(self):
        """
        For each particle, plot the evolution of each..
        """
        try:
            pass
        except Exception as e:
            traceback.print_exc()
            print(f"Error plotting particle evolution: {e}")
           
    def plot_particle_scatter(self):
        """
        For each iteration, plot a scatter plot of all particles...
        """
        try:
            pass            
        except Exception as e:
            traceback.print_exc()
            print(f"Error plotting particle scatter: {e}")
    
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
        self.plot_particle_scatter()
        self.plot_gbest_y_hist()