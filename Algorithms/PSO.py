#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2025/7/16
# @Author  : Nastaran Kianersi (based on PSO.py at github.com/guofei9987)

import matplotlib.pyplot as plt
import numpy as np
import sys

from . import utils
from .DecisionVar import *
from .InnerPSO import *
import subprocess
from multiprocessing import Process, Manager, Pipe
from io import StringIO

import time
import os
import copy

import csv
import re
from pathlib import Path
from datetime import datetime
import traceback


def parse_property_file(filepath):
    """Parse the property type mapping file.

    Each line: <comma-separated props>, <type>
    Example:
        temperature, continuous
        return_flow, discrete
        flow1, flow2, flow3, continuous
    """
    continuous, discrete = set(), set()

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue  # skip empty/commented lines

            try:
                # Split by comma and then by type (continuous/discrete)
                *props, typ = [item.strip() for item in line.split(',')]
                typ = typ.lower()
                props = [p for p in props if p]  # remove empty entries

                if typ not in ('continuous', 'discrete'):
                    print(f"Skipping line '{line}' (invalid type: {typ})")
                    continue

                target_set = continuous if typ == 'continuous' else discrete
                target_set.update(props)
            except Exception as e:
                print(f"Failed to parse line '{line}': {e}")

    return sorted(list(continuous)), sorted(list(discrete))

##### 
# Bypassing stdout_capture (commenting out old_std<sth> and sys.<sth> = old_std<sth>)
# makes the child processes print directly to the console, which is useful for 
# debugging but can be very messy.
def inner_loop_wrapper(procnum, inner_pso_obj, return_dict, *args):
    stdout_capture = StringIO()
    stderr_capture = StringIO()
    
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    
    sys.stdout = stdout_capture
    sys.stderr = stderr_capture
    
    # inner_pso = copy.deepcopy(inner_pso_obj)
    results = inner_pso_obj.optimize_distributed(*args)
    return_dict[procnum] = {
            'results': results,
            'stdout': stdout_capture.getvalue(),
            'stderr': stderr_capture.getvalue()
        }
    
    sys.stdout = old_stdout
    sys.stderr = old_stderr


def one_iter_wrapper(procnum, inner_pso_obj, return_dict, *args):
    stdout_capture = StringIO()
    stderr_capture = StringIO()
    
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    
    sys.stdout = stdout_capture
    sys.stderr = stderr_capture
    
    results = inner_pso_obj.run_inner_loop_once(*args)
    return_dict[procnum] = {
            'results': results,
            'stdout': stdout_capture.getvalue(),
            'stderr': stderr_capture.getvalue()
        }
    
    sys.stdout = old_stdout
    sys.stderr = old_stderr
##### 

class PSO():
    r"""
    PSO (Particle swarm optimization) algorithm.

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

    .. [IJCNN1995] J. Kennedy and R.C. Eberhart, "Particle Swarm Optimization,"
    Proceedings of the IEEE International Joint Conference on Neural
    Networks, 1995, pp. 1942-1948.

    This class is designed specifically for optimizing DecisionVar instances
    using a distributed approach where each particle's fitness evaluation is
    performed by submitting a separate job to a cluster via a SLURM job script.
    The value returned by each job is used as the fitness score for that particle,
    and we are optimizing for that fitness score.
    
    Parameters
    --------------------
    exec_trace_addr : str
        Path to the executable trace file required for job execution.
    configs_dir : str
        Path to the property files used in the optimization jobs.
    output_dir : str
        Directory where output files and results will be saved.
    w : float, optional (default=0.8)
        Inertia weight controlling the influence of previous velocity.
    c1 : float, optional (default=1)
        Cognitive parameter controlling the influence of personal best.
    c2 : float, optional (default=1)
        Social parameter controlling the influence of global best.
    auto_coeffs : bool, optional (default=True)
        If True, automatically update PSO coefficients during optimization.
    early_termination : bool, optional (default=False)
        If True, terminate early if convergence is detected.
    verbose : bool, optional (default=False)
        If True, print detailed information during optimization.
    
    Attributes
    ----------------------
    X : np.ndarray
        Current positions of all particles (array of DecisionVar).
    V : np.ndarray
        Current velocities of all particles.
    Y : np.ndarray
        Current fitness values of all particles.
    pbest_x : np.ndarray
        Personal best positions for each particle.
    pbest_y : np.ndarray
        Personal best fitness values for each particle.
    gbest_x : DecisionVar
        Global best position found by the swarm.
        Global best fitness value found by the swarm.
        History of global best fitness values across iterations.
    X_hist : np.ndarray
        History of particle positions for each iteration (for plotting).
    Y_hist : np.ndarray
        History of particle fitness values for each iteration (for plotting).
    save_dir : str
        Directory where run-specific outputs and plots are saved.
    record_mode : bool
        If True, records additional values for analysis.
    record_value : dict
        Stores recorded values for positions, velocities, and fitness.
    best_x : DecisionVar
        Best position found (for backward compatibility).
    best_y : float
        Best fitness value found (for backward compatibility).
    lb : np.ndarray
        Lower bounds for each DecisionVar dimension.
    ub : np.ndarray
        Upper bounds for each DecisionVar dimension.
    n_dim : int
        Number of dimensions in DecisionVar.
    V_prev_gamma : np.ndarray
        Previous velocity values for the rbf_gamma parameter (for discrete/binary PSO).
    iter_num : int
        Current iteration number.

    """
    def __init__(self, exec_trace_addr, configs_dir, output_dir, pop, max_iter,
                 w=0.8, c1=1, c2=1, auto_coeffs=True, skip_inner_loop=False, 
                 skip_event_sel_opt=False, skip_learner_opt=False, 
                 early_termination=False, verbose=False):
        
        self.exec_trace_addr = exec_trace_addr
        self.configs_dir = configs_dir
        self.output_dir = output_dir
        dt_str = datetime.now().strftime("run_%Y-%m-%d_at_%H-%M-%S")
        self.save_dir = os.path.join(self.output_dir, dt_str)
        os.makedirs(self.save_dir, exist_ok=True)
        self.iter_num = 0
        
        self.onetime_process = []
        self.onetime_job_id = []
        
        # Parse .prop files to determine the size of CompositeDecisionVar and the dim of particles in InnerPSO
        with open(os.path.join(self.configs_dir, 'key_prop.txt'), 'r') as f:
            self.key_prop = f.read().strip()
            
        _, self.onetime_props = parse_property_file(os.path.join(self.configs_dir, 'onetime_props.prop'))
        self.continuous_props, _ = parse_property_file(os.path.join(self.configs_dir, 'optimizing_props.prop'))
        for i, prop in enumerate(self.continuous_props):
            fname = os.path.join(self.configs_dir, f"optimizing_props_{i}.prop")
            with open(fname, "w") as fh:
                fh.write(f"{prop}, continuous\n")
            
        # ... for InnerPSO instances
        self.all_props = self.onetime_props + self.continuous_props
        inner_pso_pop      = 20 #15
        inner_pso_max_iter = 20 #20 #10
        # create an array of InnerPSO instances (one per particle)
        self.inner_psos = np.array(
            [InnerPSO(self.all_props, self.key_prop, inner_pso_pop, inner_pso_max_iter, 
                      skip_event_sel_opt=skip_event_sel_opt, 
                      skip_learner_opt=skip_learner_opt, early_termination=True) for _ in range(pop)],
            dtype=object
        )
        self.inner_loop_gbest_x_init = np.array([None] * pop, dtype=object)
        self.skip_inner_loop = skip_inner_loop
        # ...
        
        self.n_opt_vars = len(self.continuous_props)
        CompositeDecisionVar.set_var_names(self.continuous_props)
        self.X = np.array([CompositeDecisionVar.random(self.n_opt_vars) for _ in range(pop)])    
        self.n_dim = self.X[0].n_dim                                              # = n_opt_vars * DecisionVar.n_dim
        self.X_hist = np.empty(dtype=CompositeDecisionVar, shape=(max_iter, pop)) # for plotting
        self.Y_hist = np.empty(dtype=float, shape=(max_iter, pop))                # for plotting
        
        self.w = w  # inertia
        self.cp, self.cg = c1, c2  # parameters to control personal best, global best respectively
        self.pop = pop  # number of particles
        self.max_iter = max_iter 
        self.auto_coeffs = auto_coeffs
        self.early_termination = early_termination  # whether to terminate early if converged
        self.verbose = verbose  # print the result of each iter or not

        self.lb, self.ub = self.X[0].lb, self.X[0].ub
        assert self.n_dim == len(self.lb) == len(self.ub), 'dim == len(lb) == len(ub) is not True'
        assert np.all(self.ub > self.lb), 'upper-bound must be greater than lower-bound'

        v_high = self.ub - self.lb
        self.V = np.random.uniform(low=-v_high, high=v_high, size=(self.pop, self.n_dim))  # speed of particles
        self.Y = np.array([0] * pop)
        
        # self.pbest_x = self.X.deepcopy()                             # personal best location of every particle in history
        # for i in range(self.pop):
        #     self.pbest_x[i] = copy.deepcopy(self.X[i])
        self.pbest_x = np.array([copy.deepcopy(self.X[i]) for i in range(self.pop)])
        self.pbest_y = np.array([np.inf] * pop)                  # best image of every particle in history
        self.gbest_x = CompositeDecisionVarMean(self.pbest_x)    # global best location for all particles
        self.gbest_y = np.inf                                    # global best y for all particles
        self.gbest_y_hist = []                                   # gbest_y of every iteration
        
        self.update_gbest()
        self.update_coeffs()
        
        # record verbose values
        self.record_mode = False
        self.record_value = {'X': [], 'V': [], 'Y': []}
        self.best_x, self.best_y = self.gbest_x, self.gbest_y  # history reasons, will be deprecated
        
        # support conditional variables: if kernel in the next iter is not rbf, rbf_gamma should not change
        self.gamma_indices = list(range(3, self.n_dim, 4))
        self.V_prev_gamma = np.copy(self.V[:, self.gamma_indices])

    def update_coeffs(self):
        if self.auto_coeffs:
            t = self.iter_num
            n = self.max_iter
            self.w = (0.4/n**2) * (t - n) ** 2 + 0.4
            self.cp = -3 * t / n + 3.5
            self.cg =  3 * t / n + 0.5
            
    def run_one_time_job(self):
        job_out_file_dir = str(os.path.join(self.save_dir, ''))
        os.makedirs(job_out_file_dir, exist_ok=True)
        
        cmd = ["bash", "./submit.sh",  job_out_file_dir, './onetime_job_base.sh', './runtime_scripts/onetime_job.sh']
        subprocess.run(cmd)         # waits for the job to be submitted
        cmd = ["sbatch", "./runtime_scripts/onetime_job.sh", str(self.exec_trace_addr), str(self.configs_dir)]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) 
        self.onetime_process = proc
            
        stdout = proc.stdout.readline()
        job_id = stdout.strip().split()[-1] if stdout.strip().startswith("Submitted batch job") else None
        self.onetime_job_id = int(job_id)
        self.onetime_job_out_file = os.path.join(job_out_file_dir, f"job_{self.onetime_job_id}.out")
        print("Submitting one-time job", job_id, "for discrete variables.")
        # Job will be done when we call self.onetime_process.wait() in call_scripts()
        
    # position and velocity are handelled for 1) rbf_gamma when kernel is not rbf, and 2) discrete variable kernel
    def update_V(self):
        r1 = np.random.rand(self.pop, self.n_dim)
        r2 = np.random.rand(self.pop, self.n_dim)
        intrm1 = np.array([self.pbest_x[i].to_array() - self.X[i].to_array() for i in range(self.pop)])
        intrm2 = np.array([self.gbest_x.to_array() - self.X[i].to_array() for i in range(self.pop)])
        
        # store copies of V columns at indices 3, 7, 11, ... which correspond to rbf_gamma of each DecisionVar
        self.V_prev_gamma = np.copy(self.V[:, self.gamma_indices])
        self.V = self.w * self.V + \
                 self.cp * r1 * intrm1 + \
                 self.cg * r2 * intrm2
                
    # position is handled for 1) rbf_gamma when kernel is not rbf 2) discrete variable kernel
    def update_X(self):
        # self.X = np.array([DecisionVar.from_array(self.X[i].to_array() + self.V[i]) for i in range(self.pop)])
        X_tmp_arr = np.array([(self.X[i].to_array() + self.V[i]) for i in range(self.pop)])
        for i in range(self.pop):
            for j in range(self.n_opt_vars):
                if self.X[i].decision_vars[j].kernel != KernelType.RBF:
                    X_tmp_arr[i, self.gamma_indices[j]] = self.X[i].decision_vars[j].rbf_gamma  # keep the same rbf_gamma if kernel is not RBF
                    self.V[i, self.gamma_indices[j]] = self.V_prev_gamma[i, j]
                
        self.X = np.array([CompositeDecisionVar.from_array(X_tmp_arr[i]) for i in range(self.pop)])
                
    # Use subprocess to run the job script and get the results
    def call_scripts(self):
        """
        For each particle, submit an sbatch job using discretization_job_base.sh,
        passing the variables in self.X as command-line arguments.
        Wait for all jobs to finish and collect their output as the fitness value.
        """
        
        job_out_file_dir = str(os.path.join(self.save_dir, f'iter_{self.iter_num}', ''))
        os.makedirs(job_out_file_dir, exist_ok=True)
        job_ids = []
        processes = []
        
        for idx, x in enumerate(self.X):
            x = copy.copy(x)
            decision_vars = x.decision_vars
            job_ids_per_var = []            
            processes_per_var = []
            
            for ith_var, decisoin_var in enumerate(decision_vars):
                decisoin_var.rbf_gamma = float(decisoin_var.rbf_gamma) if decisoin_var.kernel == KernelType.RBF else 0.0
                args = [f"{key}={value.name.lower() if isinstance(value, Enum) else int(value) if isinstance(value, float) and value.is_integer() else value}"
                        for key, value in vars(decisoin_var).items()]
                
                cmd = ["bash", "./submit.sh",  job_out_file_dir, './discretization_job_base.sh', './runtime_scripts/discretization_job.sh']
                subprocess.run(cmd)
                
                cmd = ["sbatch", "--parsable", "./runtime_scripts/discretization_job.sh", str(self.exec_trace_addr), str(os.path.join(self.configs_dir, f"optimizing_props_{ith_var}.prop"))] + args 
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) # Async process
                processes_per_var.append(proc)
                
                job_id = proc.stdout.strip()
                job_ids_per_var.append(int(job_id))
                print("Submitting job", job_id, "with args: ", ' '.join(args))
                
            job_ids.append(job_ids_per_var)
            processes.append(processes_per_var)
    
        
        # Each particle submits multiple jobs (one per discrete variable)
        # Check if all jobs for a particle are finished
        # Then, call the inner loop to collect results for that particle
        inner_loop_processes = [None] * self.pop
        pending_particles = list(range(self.pop))
        
        manager = Manager()
        return_dict = manager.dict()
            
        while pending_particles:
            for idx in pending_particles:
                if len(processes[idx]) == 0:
                    # Make sure if onetime job is done (only the first time this function is called)
                    if self.onetime_process is not None:
                        self.onetime_process.wait()
                        self.onetime_process = None
            
                    cmd = ["bash", "./submit.sh",  str(os.path.join(job_out_file_dir, f"particle_{idx}", "")), './prepare_learner_input_base.sh', './runtime_scripts/prepare_learner_input.sh']
                    subprocess.run(cmd)
                
                    # All jobs in the particle are done, now call the inner loop
                    # Call an sbatch script to prepare the input to the learner (i.e., inner loop)
                    # Then call the inner loop 
                    leaner_input_foldername = "event_traces_iter_{}_particle_{}".format(self.iter_num, idx)    
                    cmd = ["sbatch", "--quiet", "./runtime_scripts/prepare_learner_input.sh", leaner_input_foldername, "onetime"] + [str(jid) for jid in job_ids[idx]]
                    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)  # waits for the job to be submitted
                    leaner_input_filename = leaner_input_foldername + ".zip"
                    
                    if self.skip_inner_loop:
                        args = (
                            idx,
                            self.inner_psos[idx],
                            return_dict,
                            os.path.join(job_out_file_dir, f"particle_{idx}"),
                            leaner_input_filename,
                            idx
                        )
                        p = Process(target=one_iter_wrapper, args=args)
                    else:
                        args = (
                            idx,
                            self.inner_psos[idx],
                            return_dict,
                            os.path.join(job_out_file_dir, f"particle_{idx}"),
                            leaner_input_filename,
                            idx,
                            self.inner_loop_gbest_x_init[idx]
                        )
                        p = Process(target=inner_loop_wrapper, args=args)
                    
                    p.start()
                    inner_loop_processes[idx] = p
                    pending_particles.remove(idx)
                else:
                    for ith_var in range(len(processes[idx])):
                        if processes[idx][ith_var] is not None and utils.is_job_completed_sacct(job_ids[idx][ith_var]):
                            # print(" ---- job ", job_ids[idx][ith_var], "is completed.")
                            processes[idx][ith_var] = None
                    
                    if all(p is None for p in processes[idx]):
                        processes[idx] = []
                                                       
            time.sleep(5)
        
        # Check for the inner loops of particles and collect their results
        pending_particles = list(range(self.pop))
        results = [0] * self.pop

        while pending_particles:
            for idx in pending_particles:
                if not inner_loop_processes[idx].is_alive():
                    print("\n----- InnerPSO for particle", idx, "has finished. Logs:")
                    try:
                        print(return_dict[idx]['stdout'], end="")
                        if return_dict[idx]['stderr']:
                            print(f"STDERR: {return_dict[idx]['stderr']}", end="")
                    except KeyError:
                        print(f"Warning: Cannot print logs for inner loop job {idx}")
                        
                    try:
                        self.inner_psos[idx] = return_dict[idx]['results']
                        results[idx] = self.inner_psos[idx].best_y
                        self.inner_loop_gbest_x_init[idx] = self.inner_psos[idx].gbest_x
                    except Exception as e:
                        print(f"Error retrieving results for particle {idx}: {e}")
                        
                    # Clean up learner input files
                    cmd = ["bash", "./cleanup.sh",  str(self.iter_num), str(idx)]
                    subprocess.run(cmd)
                        
                    pending_particles.remove(idx)
                    args_chosen = [self.all_props[i] for i in range(len(self.all_props)-1) if self.inner_psos[idx].best_x.bin_decision_var.arr[i] == 1] + [self.inner_psos[idx].event_sel_args[-1]]
                    learner_args_chosen = [val for val in self.inner_psos[idx].best_x.learner_decision_var.to_array(dtype=object)]
                    print("----- Results from particle", idx, " InnerPSO operation: inner_loop_iter", self.inner_psos[idx].iter_num, ", best score", self.inner_psos[idx].best_y, "at", args_chosen, " and ", learner_args_chosen, "-----\n")
                
            time.sleep(5)
                    
        self.Y = np.array(results)  
        return self.Y
    
    # TODO: submit a job that runs particle job with the best parameters found
    # and saves the traces and the learned model to output_dir
    def run_for_xbest(self):
        pass
    
    def update_pbest(self):
        '''
        personal best
        :return:
        '''
        self.need_update = self.pbest_y > self.Y
        # print("PSO update_pbest: ", self.need_update.shape, self.pbest_y.shape, self.Y.shape)
        # self.pbest_x = np.where(self.need_update, self.X, self.pbest_x)
        for i in range(self.pop):
            if self.need_update[i]:
                self.pbest_x[i] = copy.deepcopy(self.X[i])
        self.pbest_y = np.where(self.need_update, self.Y, self.pbest_y)

    def update_gbest(self):
        '''
        global best
        :return:
        '''
        idx_min = self.pbest_y.argmin()
        # print("PSO update_gbest: ", self.gbest_y, self.pbest_y.shape, self.pbest_y[idx_min])
        if self.gbest_y > self.pbest_y[idx_min]:
            self.gbest_x = self.X[idx_min]
            self.gbest_y = self.pbest_y[idx_min]

    def recorder(self):
        if not self.record_mode:
            return
        self.record_value['X'].append(self.X)
        self.record_value['V'].append(self.V)
        self.record_value['Y'].append(self.Y)
    
    def iterate(self, iter_num):
        converged = False
        
        self.iter_num = iter_num
        print("Iter", iter_num, "Coeffs: w =", self.w, ", cp =", self.cp, ", cg =", self.cg)
        self.update_V()
        self.recorder()
        self.update_X()
        self.call_scripts() # Call the jobs and wait for them to finish, then the returned value would be the value
                            # of y for each particle. Run a separate job for each particle.
        self.update_pbest()
        self.update_gbest()
        self.update_coeffs()
        
        if self.verbose:
            print('Iter: {}, Best fit: {} at {}'.format(iter_num, self.gbest_y, self.gbest_x))

        self.gbest_y_hist.append(self.gbest_y)
        self.Y_hist[iter_num] = self.Y.copy()
        self.X_hist[iter_num] = self.X.copy()
        
        print("PSO iteration {} completed. Global best fitness: {}".format(iter_num, self.gbest_y))
        if self.early_termination:
            if self.analyze_convergence():
                print(f"Converged at iteration {iter_num}. Stopping early.")
                converged = True
            
        return converged
    
    def optimize_distributed(self):
        self.run_one_time_job()
             
        for iter_num in range(self.max_iter):
            converged = self.iterate(iter_num)
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

        if abs(self.gbest_y) < 1e-12:
            return False
        
        recent = self.gbest_y_hist[-patience:]
        past = self.gbest_y_hist[-(patience+1)]
        
        improvement = abs(recent[-1] - past)
        scale = max(abs(past), 1.0)  # avoid division by zero
        rel_improvement = improvement / scale
        
        print("PSO gbest improvement analysis:", "converged" if rel_improvement < rel_tol else "not converged")
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
        pos_spread = np.mean(CompositeDecisionVarStd(positions).to_array())
        pos_scale = np.mean(CompositeDecisionVarMean(CompositeDecisionVarAbs(positions)).to_array()) + 1e-9
        position_converged = pos_spread / pos_scale < tol_factor
        
        # Velocity convergence: near-zero average velocity
        vel_norms = np.linalg.norm(velocities, axis=1)
        velocity_converged = np.mean(vel_norms) < tol_factor * pos_scale
        
        print("PSO Convergence analysis based on position:", "converged" if position_converged else "not converged")
        print("PSO Convergence analysis based on velocity:", "converged" if velocity_converged else "not converged")
        
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