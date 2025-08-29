#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/8/20
# @Author  : github.com/guofei9987

import numpy as np
from .base import SkoBase
from .DecisionVar import *
import subprocess
import time
import os

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
    def __init__(self, exec_trace_addr, output_dir, jobscript_addr, pop, max_iter, w=0.8, c1=0.5, c2=0.5,
                 constraint_eq=tuple(), constraint_ueq=tuple(), verbose=False):

        self.exec_trace_addr = exec_trace_addr
        self.output_dir = output_dir
        self.jobscript_addr = jobscript_addr
        
        self.X = np.array([DecisionVar.random() for _ in range(pop)])
        self.n_dim = DecisionVar.n_dim  
        
        self.w = w  # inertia
        self.cp, self.cg = c1, c2  # parameters to control personal best, global best respectively
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
        self.gbest_y_hist = []  # gbest_y of every iteration
        self.update_gbest()

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
        
    # Use subprocess to run the job script and get the results
    def call_scripts(self):
        """
        For each particle, submit an sbatch job using the jobscript at self.jobscript_addr,
        passing the variables in self.X as command-line arguments.
        Wait for all jobs to finish and collect their output as the fitness value.
        """

        processes = []
        job_ids = []
        results = [0] * self.pop
        for idx, x in enumerate(self.X):
            args = [f"{key}={value.name.lower() if isinstance(value, Enum) else int(value) if isinstance(value, float) and value.is_integer() else value}"
                    for key, value in vars(x).items()]
            cmd = ["sbatch", self.jobscript_addr, str(self.exec_trace_addr)] + args  # Also give output_dir to hpm_job_pso to copy output files to that directory.
            print("Submitting job with command:", ' '.join(cmd))
            
            # Can use subprocess.PIPE for job_id only; for the rest of stdout, read the output files instead.
            # You can use shell=True to give the whole command as a string, but it has security risks.
            # Text=true: returns buffers in text mode, not bytes
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) # waits for the job to be submitted
            processes.append(proc)
            stdout = proc.stdout.readline()
            job_id = stdout.strip().split()[-1] if stdout.strip().startswith("Submitted batch job") else None
            job_ids.append(int(job_id))
            
            # To read and print stdout and stderr in real-time (but can be blocking on stdout/stderr.readline())
            # output = ""
            # error = ""
            # while proc.poll() is None:
            #     stdout = proc.stdout.readline()
            #     print(stdout.strip())
            #     output += stdout
            #     stderr = proc.stderr.readline()
            #     print(stderr.strip())
            #     error += stderr
            
            # Old way to read stdout and stderr after process ends (if sbatch returns right after sumbission)
            # stdout, _ = proc.communicate()
            # print(stdout)
            # job_id = stdout.strip().split()[-1] if stdout.strip().startswith("Submitted batch job") else None
            # job_ids.append(int(job_id))
           
        print("Job IDs:", job_ids)
        # print("Current working directory:", os.getcwd())
        # print("Files in current directory:", os.listdir(os.getcwd()), "\n")
        
        # Check if all jobs are finished by checking their output files
        pending = list(range(self.pop))
        while pending:
            for idx in pending:
                if processes[idx].poll() is not None:  # Process finished
                    out_file = f"./job_{job_ids[idx]}.out"
                    try:
                        with open(out_file, "r") as f:
                            print(f"Reading output file {out_file}...")
                            output = f.read()
                            if "job ended" in output:
                                lines = output.strip().splitlines()
                                for line in reversed(lines):
                                    try:
                                        val = float(line)
                                        results[idx] = val
                                        break
                                    except ValueError:
                                        continue
                                pending.remove(idx)
                    except FileNotFoundError:
                        print(f"Output file {out_file} not found yet.")
                        raise FileNotFoundError(f"Output file {out_file} not found.")
                else:
                    time.sleep(10)
            
        
        print("All particle jobs finished.")
        print("Collecting results...")
        
        # TODO: collect results from the jobs, or maybe just the output of the 
        self.Y = np.array(results)  # Assuming results are already in the correct shape
        return self.Y
    
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
    def run_distributed(self, precision=None, N=20):
        '''
        precision: None or float
            If precision is None, it will run the number of max_iter steps
            If precision is a float, the loop will stop if continuous N difference between pbest less than precision
        N: int
        '''
        c = 0
        for iter_num in range(self.max_iter):
            self.update_V()
            self.recorder()
            self.update_X()
            self.call_scripts() # Call the jobs and wait for them to finish, then the returned value would be the value
                                # of y for each particle. Run a separate job for each particle.
            self.update_pbest()
            self.update_gbest()
            
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
        self.best_x, self.best_y = self.gbest_x, self.gbest_y
        return self.best_x, self.best_y

    fit_distributed = run_distributed
