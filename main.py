from Algorithms.PSO import PSO, parse_property_file
from Algorithms.InnerPSO import InnerPSO
import os
from argparse import ArgumentParser
from pathlib import Path
from Algorithms.DecisionVar import *


parser = ArgumentParser(description="Process Nibe data in nfer format to create time series and generate event traces.")
parser.add_argument("exec_trace_addr", type=Path, nargs=1, help="Path to execution trace file")
parser.add_argument("configs_dir", type=Path, nargs=1, help="Path to config files (.prop files determining variables to be optimized)")
parser.add_argument("--output-dir", default=os.getenv('PWD', os.getcwd()) + '/outputs', help="Directory for saving output files")

args = parser.parse_args()
exec_trace_addr = args.exec_trace_addr[0].resolve()
configs_dir = args.configs_dir[0].resolve()

output_dir = os.path.abspath(args.output_dir)
os.makedirs(output_dir, exist_ok=True)
print(f"Execution trace address: {exec_trace_addr}")
print(f"Output directory: {output_dir}\n\n")

## ------- InnerPSO test -------
# with open(os.path.join(configs_dir, 'key_prop.txt'), 'r') as f:
#     key_prop = f.read().strip()
    
# _, onetime_props = parse_property_file(os.path.join(configs_dir, 'onetime_props.prop'))
# continuous_props, _ = parse_property_file(os.path.join(configs_dir, 'optimizing_props.prop'))
# for i, prop in enumerate(continuous_props):
#     fname = os.path.join(configs_dir, f"optimizing_props_{i}.prop")
#     with open(fname, "w") as fh:
#         fh.write(f"{prop}, continuous\n")
    
# print("onetime props:", onetime_props)
# print("continuous props:", continuous_props)
# print("key prop:", key_prop, "\n\n")

# all_props = onetime_props + continuous_props
# inner_pso_pop      = 10 #30
# inner_pso_max_iter = 20 #50
# inner_psos = InnerPSO(all_props, key_prop, inner_pso_pop, inner_pso_max_iter)

# inner_psos.optimize_distributed('./', "event_traces.zip")
## -------------------------


# exec_trace_addr is the input directory
# output_dir is the output directory
# jobscript_addr is the job script address, can have a different path but read from pwd for now
# (of course it will have some effect on the exec time cause the file has to be loaded onto memmory)
pso = PSO(exec_trace_addr=exec_trace_addr, configs_dir=configs_dir, output_dir=output_dir,
          pop=4, max_iter=5, auto_coeffs=True, skip_inner_loop=False, 
          skip_event_sel_opt=False, skip_learner_opt=False, early_termination=True) #pop=128, max_iter=100   # pop=30, max_iter=30
best_x, best_y = pso.optimize_distributed()
pso.analyze_convergence()
print("** Best solution: **")
print("Best objective value:", best_y)
print("-- Outer loop best_x:", best_x)
print("-- Inner loop best_x:")
print("Learner Params:", pso.inner_loop_gbest_x_init[pso.gbest_y_idx].learner_decision_var)
indices = np.where(pso.inner_loop_gbest_x_init[pso.gbest_y_idx].bin_decision_var.arr == 1)
chosen_events = list(np.array(pso.inner_psos[0].event_sel_args)[indices[0]]) + [pso.inner_psos[0].event_sel_args[-1]]
print("Chosen events:", chosen_events)

# pso.visualize()
pso.run_for_xbest()

