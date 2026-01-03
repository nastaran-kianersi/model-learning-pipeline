import numpy as np

from enum import Enum
import itertools
import operator


class BinaryDecisionVar:
    def __init__(self, n_dim):
        self.n_dim = n_dim
        # self.arr = np.random.randint(0, 2, size=n_dim)  # Random binary values (0 or 1)
        self.arr = np.random.choice([0, 1], size=n_dim)
    
    @staticmethod
    def random(n_dim):
        return BinaryDecisionVar(n_dim)
    
    @classmethod
    def from_array(cls, arr):
        instance = cls(len(arr))
        instance.arr = arr
        return instance
    
    def __repr__(self):
        return f"BinaryDecisionVar(arr={self.arr})"
    
    
class KernelType(Enum):
    LINEAR = 0
    RBF = 1
    COSINE = 2

    # Makes sense for only two operands; can't be used to sum up more than
    # two KernelType instances because it does not have the commutative property 
    def __add__(self, other):
        if not isinstance(other, KernelType):
            return NotImplemented
        
        if self == other:
            return self
        
        num = (self.value + other.value) % len(KernelType)
        return KernelType(num)

    # Makes sense for only two operands; can't be used to sum up more than
    # two KernelType instances because it does not have the commutative property
    def __sub__(self, other):
        if not isinstance(other, KernelType):
            return NotImplemented
        
        if self == other:
            return self
        
        num = (self.value - other.value) % len(KernelType)
        return KernelType(num)

""" 
This class can easily turn into an abstract base class (interface) if needed
Methods/vars to be implemented:
    - n_dim: Number of dimensions (static attribute)
    - lb: Lower bounds (static attribute)
    - ub: Upper bounds (static attribute)
    - Customize: add static member variables which represent the values to be optimized
    - random(): Generate a random instance of DecisionVar
    - to_array(): Convert to a numpy array
    - clip(): Ensure values are within bounds
    - + and - operators for arithmetic operations on DecisionVar instances
    - __eq__(): Equality check for DecisionVar instances
    - __repr__(): String representation of the instance
    - __init__(min_size, penalty, kernel, rbf_gamma): Initialize with specific values
    - from_array(arr): Initialize from a numpy array
"""
class DecisionVar:
    n_dim = 4
    lb = np.array([50, 5.0, 0, 0.05])
    ub = np.array([1000, 500.0, len(KernelType) - 1, 10.0])
        
    # in update_X and update_V, handle rbf_gamma when kernel is not rbf,
    # in here, handle discrete arithmetic
    @staticmethod
    def random():
        min_size = np.random.randint(DecisionVar.lb[0], DecisionVar.ub[0])
        penalty = np.random.uniform(DecisionVar.lb[1], DecisionVar.ub[1])
        kernel = np.random.choice(list(KernelType))
        rbf_gamma = np.random.uniform(DecisionVar.lb[3], DecisionVar.ub[3])
        return DecisionVar(min_size, penalty, kernel, rbf_gamma)
    
    def __init__(
        self,
        min_size: int,
        penalty: float,
        kernel: KernelType,
        rbf_gamma: float,
    ):
        self.min_size = min_size
        self.penalty = penalty
        self.kernel = kernel
        self.rbf_gamma = rbf_gamma
        self.clip()  # Ensure values are within bounds
        
    @classmethod
    def from_array(cls, arr):
        """Initialize DecisionVar from a numpy array."""
        if len(arr) != DecisionVar.n_dim:
            raise ValueError(f"Array must have {DecisionVar.n_dim} elements.")
        
        return cls(
            min_size=int(arr[0]),
            penalty=float(arr[1]),
            kernel=KernelType(int(np.clip(arr[2], 0, len(KernelType) - 1))),
            rbf_gamma=float(arr[3])
        )
        
    def __repr__(self):
        return (f"DecisionVar(min_size={self.min_size}, penalty={self.penalty}, "
                f"kernel={self.kernel}, rbf_gamma={self.rbf_gamma})")
        
    # The way arithmetic operations are defined for KernelType might impact convergence
    def __add__(self, other):
        if not isinstance(other, DecisionVar):
            return NotImplemented
        
        arr_sum = self.to_array() + other.to_array()
        kernel = self.kernel + other.kernel
            
        result = DecisionVar(
            min_size=int(arr_sum[0]),
            penalty=float(arr_sum[1]),
            kernel=kernel,
            rbf_gamma=float(arr_sum[3])
        )
        return result

    # The way arithmetic operations are defined for KernelType might impact convergence
    def __sub__(self, other):
        if not isinstance(other, DecisionVar):
            return NotImplemented
        
        arr_diff = self.to_array() - other.to_array()
        kernel = self.kernel - other.kernel
        
        result = DecisionVar(
            min_size=int(arr_diff[0]),
            penalty=float(arr_diff[1]),
            kernel=kernel,
            rbf_gamma=float(arr_diff[3])
        )
        return result
    
    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, DecisionVar):
            return self.min_size == other.min_size and \
                   self.penalty == other.penalty and \
                   self.kernel == other.kernel and \
                   self.rbf_gamma == other.rbf_gamma
        return False

    # Instead of __array__; to avoid implicit conversion to numpy array
    
    def to_array(self, dtype=None):
        vals = [
            self.min_size,
            self.penalty,
            int(self.kernel.value),
            self.rbf_gamma,
        ]
        return np.array(vals, dtype=dtype)
    
    def clip(self):
        """Clip the values of the DecisionVar instance to its bounds."""
        self.min_size = np.clip(self.min_size, DecisionVar.lb[0], DecisionVar.ub[0])
        self.penalty = np.clip(self.penalty, DecisionVar.lb[1], DecisionVar.ub[1])
        self.kernel = KernelType(int(np.clip(self.kernel.value, DecisionVar.lb[2], DecisionVar.ub[2])))
        self.rbf_gamma = np.clip(self.rbf_gamma, DecisionVar.lb[3], DecisionVar.ub[3])
    

# Wrapper around multiple DecisionVar instances
class CompositeDecisionVar:
    @staticmethod
    def random(n_vars):
        decision_vars = [DecisionVar.random() for _ in range(n_vars)]
        return CompositeDecisionVar(decision_vars)
        
    def __init__(self, decision_vars):
        self.n_dim_per_var = DecisionVar.n_dim
        self.n_vars = len(decision_vars)                    
        self.n_dim = self.n_dim_per_var * self.n_vars
        self.decision_vars = decision_vars
        self.lb = np.tile(DecisionVar.lb, self.n_vars)
        self.ub = np.tile(DecisionVar.ub, self.n_vars)
    
    # convert from fully flattened array
    @classmethod
    def from_array(cls, arr):
        """Initialize CompositeDecisionVar from a numpy array."""
        n_dim_per_var = DecisionVar.n_dim
        if len(arr) % n_dim_per_var != 0:
            raise ValueError(f"Array length must be a multiple of {n_dim_per_var}.")
        
        decision_vars = []
        for i in range(0, len(arr), n_dim_per_var):
            var_array = arr[i:i + n_dim_per_var]
            decision_var = DecisionVar.from_array(var_array)
            decision_vars.append(decision_var)
        
        return cls(decision_vars)
    
    def __repr__(self):
        return f"CompositeDecisionVar(n_vars={self.n_vars}, decision_vars={self.decision_vars})"
    
    def __add__(self, other):
        if not isinstance(other, CompositeDecisionVar):
            return NotImplemented
        
        if self.n_vars != other.n_vars:
            raise ValueError("Both CompositeDecisionVar instances must have the same number of DecisionVar instances.")
        
        new_decision_vars = [
            self.decision_vars[i] + other.decision_vars[i]
            for i in range(self.n_vars)
        ]
        
        return CompositeDecisionVar(new_decision_vars)
    
    def __sub__(self, other):
        if not isinstance(other, CompositeDecisionVar):
            return NotImplemented
        
        if self.n_vars != other.n_vars:
            raise ValueError("Both CompositeDecisionVar instances must have the same number of DecisionVar instances.")
        
        new_decision_vars = [
            self.decision_vars[i] - other.decision_vars[i]
            for i in range(self.n_vars)
        ]
        
        return CompositeDecisionVar(new_decision_vars)
    
    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, CompositeDecisionVar):
            return all(self.decision_vars[i] == other.decision_vars[i] for i in range(self.n_vars))
        return False
    
    # Instead of __array__; to avoid implicit conversion to numpy array
    # fully flatten: convert each decision var to array and concatenate (turn to 1d array)
    def to_array(self, dtype=None):
        arrays = [var.to_array(dtype=dtype) for var in self.decision_vars]
        return np.concatenate(arrays)
    
    def clip(self):
        """Clip the values of all DecisionVar instances to their bounds."""
        for var in self.decision_vars:
            var.clip()
    
    
# =========================
# Helper functions
# =========================

def most_common(L):
  # get an iterable of (item, iterable) pairs
  SL = sorted((x, i) for i, x in enumerate(L))
  # print 'SL:', SL
  groups = itertools.groupby(SL, key=operator.itemgetter(0))
  # auxiliary function to get "quality" for an item
  def _auxfun(g):
    item, iterable = g
    count = 0
    min_index = len(L)
    for _, where in iterable:
      count += 1
      min_index = min(min_index, where)
    # print 'item %r, count %r, minind %r' % (item, count, min_index)
    return count, -min_index
  # pick the highest-count/earliest item
  return max(groups, key=_auxfun)[0]


def DecisionVarMean(decision_vars):
    """Calculate the mean of a list of DecisionVar instances."""
    if len(decision_vars) == 0:
        return None
        
    values = [var.to_array() for var in decision_vars]    
    mean_values = np.mean(values, axis=0)
    '''For kernel, use the most common value (mode)
       If more than kernel type is common, choose the first one in the list'''
    kernel_types = [int(var.kernel.value) for var in decision_vars]
    kernel_type_mode = most_common(kernel_types)    # max(set(kernel_types), key=kernel_types.count)
    mean_values[2] = kernel_type_mode
    
    return DecisionVar(
        min_size=int(mean_values[0]),
        penalty=float(mean_values[1]),
        kernel=KernelType(int(mean_values[2])),
        rbf_gamma=float(mean_values[3])
    )
    
    
def DecisionVarStd(decision_vars):
    """Calculate the standard deviation of a list of DecisionVar instances."""
    if len(decision_vars) == 0:
        return None
        
    values = np.array([var.to_array() for var in decision_vars])
    std_values = np.std(values, axis=0)
    
    '''For kernel, use the most common value (mode)
    If more than kernel type is common, choose the first one in the list'''
    kernel_types = [int(var.kernel.value) for var in decision_vars]
    kernel_type_mode = most_common(kernel_types)    # max(set(kernel_types), key=kernel_types.count)
    std_values[2] = kernel_type_mode
    
    return DecisionVar(
        min_size=int(std_values[0]),
        penalty=float(std_values[1]),
        kernel=KernelType(int(std_values[2])),
        rbf_gamma=float(std_values[3])
    )
    
    
def DecisionVarAbs(decision_vars):
    """Calculate the abs of a list of DecisionVar instances."""
    if len(decision_vars) == 0:
        return None
        
    ### The generic way:
    # values = [var.to_array() for var in decision_vars]    
    # abs_vals = np.abs(values)
    # abs_values = np.array([DecisionVar.from_array(abs_val) for abs_val in abs_vals])
    # return abs_values
    
    return decision_vars # since all values are positive already
    
    
def CompositeDecisionVarMean(composite_decision_vars):
    """Calculate the mean of a list of CompositeDecisionVar instances."""
    if len(composite_decision_vars) == 0:
        return None
        
    n_vars = composite_decision_vars[0].n_vars
    decision_vars_per_position = [[] for _ in range(n_vars)]
    
    for comp_var in composite_decision_vars:
        for i in range(n_vars):
            decision_vars_per_position[i].append(comp_var.decision_vars[i])
    
    mean_decision_vars = [
        DecisionVarMean(decision_vars_per_position[i])
        for i in range(n_vars)
    ]
    
    return CompositeDecisionVar(mean_decision_vars)


def CompositeDecisionVarStd(composite_decision_vars):
    """Calculate the standard deviation of a list of CompositeDecisionVar instances."""
    if len(composite_decision_vars) == 0:
        return None
        
    n_vars = composite_decision_vars[0].n_vars
    decision_vars_per_position = [[] for _ in range(n_vars)]
    
    for comp_var in composite_decision_vars:
        for i in range(n_vars):
            decision_vars_per_position[i].append(comp_var.decision_vars[i])
    
    std_decision_vars = [
        DecisionVarStd(decision_vars_per_position[i])
        for i in range(n_vars)
    ]
    
    return CompositeDecisionVar(std_decision_vars)


def CompositeDecisionVarAbs(composite_decision_vars):
    """Calculate the abs of a list of CompositeDecisionVar instances."""
    if len(composite_decision_vars) == 0:
        return None
        
    # n_vars = composite_decision_vars[0].n_vars
    # decision_vars_per_position = [[] for _ in range(n_vars)]
    
    # for comp_var in composite_decision_vars:
    #     for i in range(n_vars):
    #         decision_vars_per_position[i].append(comp_var.decision_vars[i])
    
    # abs_decision_vars = [
    #     DecisionVarAbs(decision_vars_per_position[i])
    #     for i in range(n_vars)
    # ]
    
    # return CompositeDecisionVar(abs_decision_vars)
    return composite_decision_vars  # since all values are positive already

# =========================
# Tests
# =========================

def Test_DecisionVarMean():
    """Test the DecisionVar class and its methods."""
    
    dv1 = DecisionVar.random()
    dv2 = DecisionVar.random()
    
    print("Random DecisionVars:")
    print(dv1)
    print(dv2)
    
    mean_dv = DecisionVarMean([dv1, dv2])
    print("Mean DecisionVar:")
    print(mean_dv)
    
    assert isinstance(mean_dv, DecisionVar), "Mean should be a DecisionVar instance"
    
    
def Test_DecisionVarArithmetic():
    """Test arithmetic operations on DecisionVar instances."""
    
    print("Testing mean, abs, and std:")
    rand_dvs = np.array([DecisionVar.random() for _ in range(10)])
    print(rand_dvs)
    print("\n\n\n\n")
    print(DecisionVarMean(rand_dvs))
    print(DecisionVarAbs(rand_dvs))
    print(DecisionVarStd(rand_dvs))
    
    dv1 = DecisionVar.random()
    dv2 = DecisionVar.random()
    
    print("Random DecisionVars:")
    print(dv1)
    print(dv2)
    
    sum_dv = dv1 + dv2
    diff_dv = dv1 - dv2
    
    print("Sum of DecisionVars:")
    print(sum_dv)
    
    print("Difference of DecisionVars:")
    print(diff_dv)
    
    print("Orignal DecisionVars:")
    print(dv1)
    print(dv2)
    
    assert isinstance(sum_dv, DecisionVar), "Sum should be a DecisionVar instance"
    assert isinstance(diff_dv, DecisionVar), "Difference should be a DecisionVar instance"
    