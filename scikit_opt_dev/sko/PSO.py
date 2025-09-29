#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/8/20
# @Author  : github.com/guofei9987 - Modified by Nastaran Kianersi

import matplotlib.pyplot as plt
import numpy as np
from .base import SkoBase
from .DecisionVar import *
import subprocess
import time
import os

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


class PSO(SkoBase):
    """
    Do PSO (Particle swarm optimization) algorithm.

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

    .. [IJCNN1995] J. Kennedy and R.C. Eberhart, "Particle Swarm Optimization,"
    Proceedings of the IEEE International Joint Conference on Neural
    Networks, 1995, pp. 1942-1948.

    Parameters
    --------------------
    func : function
        The func you want to do optimal
    dim : int
        Number of dimension, which is number of parameters of func.
    pop : int
        Size of population, which is the number of Particles. We use 'pop' to keep accordance with GA
    max_iter : int
        Max of iter iterations
    lb : array_like
        The lower bound of every variables of func
    ub : array_like
        The upper bound of every variables of func
    constraint_eq : tuple
        equal constraint. Note: not available yet.
    constraint_ueq : tuple
        unequal constraint
    n_processes : int
        Number of processes, 0 means use all cpu
    Attributes
    ----------------------
    pbest_x : array_like, shape is (pop,dim)
        best location of every particle in history
    pbest_y : array_like, shape is (pop,1)
        best image of every particle in history
    gbest_x : array_like, shape is (1,dim)
        general best location for all particles in history
    gbest_y : float
        general best image  for all particles in history
    gbest_y_hist : list
        gbest_y of every iteration


    Examples
    -----------------------------
    see https://scikit-opt.github.io/scikit-opt/#/en/README?id=_3-psoparticle-swarm-optimization
    """

    
    # For later: Use inheritance and use the base class in __init__
    def __init__(self, exec_trace_addr, output_dir, jobscript_addr, pop, max_iter, w=0.8, c1=1, c2=1,
                 auto_coeffs=True, constraint_eq=tuple(), constraint_ueq=tuple(), verbose=False):

        self.exec_trace_addr = exec_trace_addr
        self.output_dir = output_dir
        self.jobscript_addr = jobscript_addr
        dt_str = datetime.now().strftime("run_%Y-%m-%d_at_%H-%M-%S")
        self.save_dir = os.path.join(self.output_dir, dt_str)
        os.makedirs(self.save_dir, exist_ok=True)
        self.iter_num = 0
        
        self.X = np.array([DecisionVar.random() for _ in range(pop)])
        self.n_dim = DecisionVar.n_dim  
        self.X_hist = np.empty(dtype=DecisionVar, shape=(max_iter, pop)) # for plotting
        self.Y_hist = np.empty(dtype=float, shape=(max_iter, pop))       # for plotting
        
        self.w = w  # inertia
        self.cp, self.cg = c1, c2  # parameters to control personal best, global best respectively
        self.auto_coeffs = auto_coeffs
        self.pop = pop  # number of particles
        self.max_iter = max_iter  # max iter
        self.verbose = verbose  # print the result of each iter or not

        self.lb, self.ub = DecisionVar.lb, DecisionVar.ub
        assert self.n_dim == len(self.lb) == len(self.ub), 'dim == len(lb) == len(ub) is not True'
        assert np.all(self.ub > self.lb), 'upper-bound must be greater than lower-bound'

        self.has_constraint = bool(constraint_ueq)
        self.constraint_ueq = constraint_ueq
        self.is_feasible = np.array([True] * pop)

        v_high = self.ub - self.lb
        self.V = np.random.uniform(low=-v_high, high=v_high, size=(self.pop, self.n_dim))  # speed of particles
        self.Y = np.array([0] * pop)
        self.pbest_x = self.X.copy()  # personal best location of every particle in history
        self.pbest_y = np.array([np.inf] * pop)  # best image of every particle in history
        self.gbest_x = DecisionVarMean(self.pbest_x)  # global best location for all particles
        self.gbest_y = np.inf  # global best y for all particles
        self.gbest_y_hist = []  # gbest_y of every iteration - not used
        self.update_gbest()
        self.update_coeffs()

        # record verbose values
        self.record_mode = False
        self.record_value = {'X': [], 'V': [], 'Y': []}
        self.best_x, self.best_y = self.gbest_x, self.gbest_y  # history reasons, will be deprecated

    def check_constraint(self, x):
        # gather all unequal constraint functions
        for constraint_func in self.constraint_ueq:
            if constraint_func(x) > 0:
                return False
        return True

    # numpy vectorization can also be used instead of list comprehension
    def update_V(self):
        r1 = np.random.rand(self.pop, self.n_dim)
        r2 = np.random.rand(self.pop, self.n_dim)
        intrm1 = np.array([(self.pbest_x[i].to_array() - self.X[i].to_array()) for i in range(self.pop)])
        intrm2 = np.array([(self.gbest_x.to_array() - self.X[i].to_array()) for i in range(self.pop)])
        self.V = self.w * self.V + \
                 self.cp * r1 * intrm1 + \
                 self.cg * r2 * intrm2

    # So not optimized
    def update_X(self):
        self.X = np.array([DecisionVar.from_array(self.X[i].to_array() + self.V[i]) for i in range(self.pop)])
        
    def update_coeffs(self):
        if self.auto_coeffs:
            t = self.iter_num
            n = self.max_iter
            self.w = (0.4/n**2) * (t - n) ** 2 + 0.4
            self.cp = -3 * t / n + 3.5
            self.cg =  3 * t / n + 0.5
            
    # Use subprocess to run the job script and get the results
    def call_scripts(self):
        """
        For each particle, submit an sbatch job using the jobscript at self.jobscript_addr,
        passing the variables in self.X as command-line arguments.
        Wait for all jobs to finish and collect their output as the fitness value.
        """
        
        job_out_file_dir = str(os.path.join(self.save_dir, f'Iter_{self.iter_num}', ''))
        os.makedirs(job_out_file_dir, exist_ok=True)
        processes = []
        job_ids = []
        results = [0] * self.pop
        
        for idx, x in enumerate(self.X):
            args = [f"{key}={value.name.lower() if isinstance(value, Enum) else int(value) if isinstance(value, float) and value.is_integer() else value}"
                    for key, value in vars(x).items()]
            cmd = ["bash", "./submit.sh",  job_out_file_dir, self.jobscript_addr, str(self.exec_trace_addr)] + args
            
            # Can use subprocess.PIPE for job_id only; for the rest of stdout, read the output files instead.
            # You can use shell=True to give the whole command as a string, but it has security risks.
            # Text=true: returns buffers in text mode, not bytes
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) # waits for the job to be submitted
            cmd = ["sbatch", "./jobscript.sh", str(self.exec_trace_addr)] + args 
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) 
            processes.append(proc)
            
            stdout = proc.stdout.readline()
            job_id = stdout.strip().split()[-1] if stdout.strip().startswith("Submitted batch job") else None
            job_ids.append(int(job_id))
            print("Submitting job", job_id, "with args: ", ' '.join(args))
            
            # Old way to read stdout and stderr after process ends (if sbatch returns right after sumbission)
            # stdout, _ = proc.communicate()
            # print(stdout)
            # job_id = stdout.strip().split()[-1] if stdout.strip().startswith("Submitted batch job") else None
            # job_ids.append(int(job_id))
           
        print("Job IDs:", job_ids)
        
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
                    results[idx] = -avg # if n > 0 else float('inf')  # Minimize negative score (maximize score)
                else:
                    time.sleep(5)
            
        self.Y = np.array(results)  
        return self.Y
    
    # TODO: submit a job that runs particle_job with the best parameters found
    # and saves the traces and the learned model to output_dir
    def run_for_xbest(self):
        pass
    
    def update_pbest(self):
        '''
        personal best
        :return:
        '''
        self.need_update = self.pbest_y > self.Y
        for idx, x in enumerate(self.X):
            if self.need_update[idx]:
                self.need_update[idx] = self.check_constraint(x)

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
    
    # finds the global miminum, so define the function such that it returns the negative of accuray
    def optimize_distributed(self, precision=None, N=20):
        '''
        precision: None or float
            If precision is None, it will run the number of max_iter steps
            If precision is a float, the loop will stop if continuous N difference between pbest less than precision
        N: int
        '''
        c = 0
        for iter_num in range(self.max_iter):
            self.iter_num = iter_num
            print("Iter ", iter_num, "Coeffs: w=", self.w, " cp=", self.cp, " cg=", self.cg)
            self.update_V()
            self.recorder()
            self.update_X()
            self.call_scripts() # Call the jobs and wait for them to finish, then the returned value would be the value
                                # of y for each particle. Run a separate job for each particle.
            self.update_pbest()
            self.update_gbest()
            self.update_coeffs()
            
            if precision is not None:
                tor_iter = np.amax(self.pbest_y) - np.amin(self.pbest_y)
                if tor_iter < precision:
                    c = c + 1
                    if c > N:
                        break
                else:
                    c = 0
            if self.verbose:
                print('Iter: {}, Best fit: {} at {}'.format(iter_num, self.gbest_y, self.gbest_x))

            self.gbest_y_hist.append(self.gbest_y)
            self.Y_hist[iter_num] = self.Y.copy()
            self.X_hist[iter_num] = self.X.copy()
            
            if self.analyze_convergence():
                print(f"Converged at iteration {iter_num}. Stopping early.")
                break
            
        self.best_x, self.best_y = self.gbest_x, self.gbest_y
        return self.best_x, self.best_y
    
    def analyze_convergence(self):
        try:
            gbest_converged = self.has_converged_gbest()
            pos_converged, vel_converged = self.has_converged_particles()
        except Exception as e:
            traceback.print_exc()
            print(f"Error in convergence analyzer: {e}")
            
        return gbest_converged and (pos_converged or vel_converged)
        
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
        pos_spread = DecisionVarMean(DecisionVarStd(positions))
        pos_scale = DecisionVarMean(DecisionVarAbs(positions)) + 1e-9
        position_converged = pos_spread / pos_scale < tol_factor
        
        # Velocity convergence: near-zero average velocity
        vel_norms = np.linalg.norm(velocities, axis=1)
        velocity_converged = np.mean(vel_norms) < tol_factor * pos_scale
        
        print("Convergence analysis based on position:", "converged" if position_converged else "not converged")
        print("Convergence analysis based on velocity:", "converged" if velocity_converged else "not converged")
        
        return position_converged, velocity_converged
    
    def plot_particle_evolution(self):
        """
        For each particle, plot the evolution of each DecisionVar field over iterations.
        Each particle gets its own figure with four subplots (one per parameter).
        Plots are saved to save_dir.
        """
        try:
            save_dir = os.path.join(self.save_dir, "individual_particle_evolution")
            os.makedirs(save_dir, exist_ok=True)
            fields = list(vars(self.X[0]).keys())
            fields.append('metric')

            for i in range(self.pop):
                fig, axs = plt.subplots(len(fields), 1, figsize=(8, 20))
                axs = axs.flatten()
                for j, field in enumerate(fields):
                    if field == 'metric':
                        y_vals = self.Y_hist[:, i]
                    elif field == 'kernel':
                        y_vals = [getattr(self.X_hist[k, i], field).value for k in range(self.max_iter)]
                    else:
                        y_vals = [getattr(self.X_hist[k, i], field) for k in range(self.max_iter)]

                    axs[j].plot(range(self.max_iter), y_vals, marker='o')
                    axs[j].set_title(f'{field} (Particle {i})')
                    axs[j].set_xlabel('Iteration')
                    axs[j].set_ylabel(field)
                    axs[j].grid()
                plt.tight_layout()
                plt.suptitle(f'Particle {i} Parameter Evolution', y=1.02)
                plt.savefig(os.path.join(save_dir, f'particle_{i}_evolution.png'), bbox_inches='tight')
                plt.close(fig)
        except Exception as e:
            traceback.print_exc()
            print(f"Error plotting particle evolution: {e}")
           
    def plot_particle_scatter(self):
        """
        For each iteration, plot a scatter plot of all particles for each DecisionVar field.
        Each iteration gets its own figure with four subplots (one per parameter).
        Plots are saved to save_dir.
        """
        try:
            save_dir = os.path.join(self.save_dir, "particle_scatterplots")
            os.makedirs(save_dir, exist_ok=True)
            fields = list(vars(self.X[0]).keys()) # Y value
            fields.append('metric')
            
            for iter_num in range(self.max_iter):
                fig, axs = plt.subplots(len(fields), 1, figsize=(8, 20))
                axs = axs.flatten()
                for j, field in enumerate(fields):
                    if field == 'metric':
                        values = self.Y_hist[iter_num, :]
                    elif field == 'kernel':
                        values = [getattr(self.X_hist[iter_num, i], field).value for i in range(self.pop)]
                    else:
                        values = [getattr(self.X_hist[iter_num, i], field) for i in range(self.pop)]

                    axs[j].scatter(range(1, self.pop + 1), values, c='b', alpha=0.7)
                    axs[j].set_title(f'{field} (Iteration {iter_num})')
                    axs[j].set_xlabel('Particle')
                    axs[j].set_ylabel(field)
                    axs[j].grid()
                plt.tight_layout()
                plt.suptitle(f'Particle Parameter Scatter (Iteration {iter_num})', y=1.02)
                plt.savefig(os.path.join(save_dir, f'iteration_{iter_num}_scatter.png'), bbox_inches='tight')
                plt.close(fig)            
        except Exception as e:
            traceback.print_exc()
            print(f"Error plotting particle scatter: {e}")
    
    def plot_gbest_y_hist(self):
        """
        Plot gbest_y_hist which is the best y value among all particles in each iteration.
        """
        try:
            _, ax = plt.subplots(figsize=(12, 8))
            plt.plot(self.gbest_y_hist, color='darkblue', linewidth=1.5)
            ax.set_title(f'Best Global y Value', pad=15) 
            ax.set_xlabel('Iter', labelpad=10)
            ax.set_ylabel('gbest_gloal', labelpad=10)
            ax.tick_params(axis='both', which='major')
            
            figure_name = f'{self.save_dir}/gbest_y_hist.png'
            plt.savefig(figure_name)
            plt.close()
        except Exception as e:
            traceback.print_exc()
            print(f"Error plotting evolution: {e}")
        
    def visualize(self):
        self.plot_particle_evolution()
        self.plot_particle_scatter()
        self.plot_gbest_y_hist()