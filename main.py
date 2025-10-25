from scikit_opt_dev.sko.PSO import PSO
import os
from argparse import ArgumentParser
from pathlib import Path
from scikit_opt_dev.sko.DecisionVar import Test_DecisionVarArithmetic as dv_test


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

# exec_trace_addr is the input directory
# output_dir is the output directory
# jobscript_addr is the job script address, can have a different path but read from pwd for now
# (of course it will have some effect on the exec time cause the file has to be loaded onto memmory)
pso = PSO(exec_trace_addr=exec_trace_addr, configs_dir=configs_dir, output_dir=output_dir,
          pop=70, max_iter=100, w=0.8, c1=1, c2=1, auto_coeffs=True, early_termination=True)
best_x, best_y = pso.optimize_distributed()
pso.analyze_convergence()
print("Best solution:", best_x)
print("Best objective value:", best_y)

pso.visualize()
pso.run_for_xbest()

