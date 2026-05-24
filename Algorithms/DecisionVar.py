import numpy as np

from enum import Enum
import itertools
import operator


class BinaryDecisionVar:
    def __init__(self, n_dim, mapping=None):
        self.n_dim = n_dim
        self.mapping = mapping
        # self.arr = np.random.randint(0, 2, size=n_dim)  # Random binary values (0 or 1)
        self.arr = np.random.choice([0, 1], size=n_dim)
    
    @staticmethod
    def random(n_dim, mapping=None):
        return BinaryDecisionVar(n_dim, mapping=mapping)
    
    @classmethod
    def from_array(cls, arr, mapping=None):
        instance = cls(len(arr), mapping=mapping)
        instance.arr = np.array(arr, copy=True)   # might be invalid! but we don't care
        instance.mapping = mapping
        return instance
    
    def __repr__(self):
        if self.mapping is not None:
            active_indices = np.where(self.arr == 1)[0]
            active_events = [self.mapping[i] for i in active_indices]
            return f"BinaryDecisionVar(arr={self.arr}, active_events={active_events})"
        else:
            return f"BinaryDecisionVar(arr={self.arr})"
    
    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, BinaryDecisionVar):
            return np.array_equal(self.arr, other.arr)
        return False
    
    
class LearnerDecisionVar:
    n_dim = 5
    lb = np.array([0, 0, 0, 0, 0]) # Interpret 0 as infinity (Null) in the learner script
    ub = np.array([1, 1, 1, 1, 10]) # np.iinfo(np.int64).max
    # b_trunc = int(np.trunc(b)), b_ceil = int(np.ceil(b))
    
    @staticmethod
    def random():
        timing_eps = np.random.uniform(LearnerDecisionVar.lb[0], LearnerDecisionVar.ub[0])
        timing_eps_decay = np.random.uniform(LearnerDecisionVar.lb[1], LearnerDecisionVar.ub[1])
        freq_eps = np.random.uniform(LearnerDecisionVar.lb[2], LearnerDecisionVar.ub[2])
        freq_eps_decay = np.random.uniform(LearnerDecisionVar.lb[3], LearnerDecisionVar.ub[3])
        tail_length = np.random.randint(LearnerDecisionVar.lb[4], LearnerDecisionVar.ub[4])
        return LearnerDecisionVar(timing_eps, timing_eps_decay, freq_eps, freq_eps_decay, tail_length)
    
    def __init__(
        self,
        timing_eps: float,
        timing_eps_decay: float,
        freq_eps: float,
        freq_eps_decay: float,
        tail_length: int,
    ):
        self.timing_eps = timing_eps
        self.timing_eps_decay = timing_eps_decay
        self.freq_eps = freq_eps
        self.freq_eps_decay = freq_eps_decay
        self.tail_length = tail_length
        # self.clip()  # Ensure values are within bounds
        
    @classmethod
    def from_array(cls, arr):
        """Initialize LearnerDecisionVar from a numpy array."""
        if len(arr) != LearnerDecisionVar.n_dim:
            raise ValueError(f"Array must have {LearnerDecisionVar.n_dim} elements.")
        
        return cls(
            timing_eps=float(arr[0]),
            timing_eps_decay=float(arr[1]),
            freq_eps=float(arr[2]),
            freq_eps_decay=float(arr[3]),
            tail_length=int(arr[4])
        )
        
    def __repr__(self):
        return (f"LearnerDecisionVar(timing_eps={self.timing_eps:.3f}, timing_eps_decay={self.timing_eps_decay:.3f}, "
                f"freq_eps={self.freq_eps:.3f}, freq_eps_decay={self.freq_eps_decay:.3f}, tail_length={self.tail_length})")
        
    # The definition of __add__ and __sub__ directly determines the
    # way boundary conditions are handled: use the + operator for
    # clamping to bounds, and use other addition functions for other
    # bounary handling methods (e.g., wrapping around)
    # ====> ORRRR since I removed self.clip() in __init__, I can just
    #        use + and - operators for normal arithmetic, and handle
    #        boundary conditions separately
    ######
    def __add__(self, other):
        if not isinstance(other, LearnerDecisionVar):
            return NotImplemented
        
        arr_sum = self.to_array() + other.to_array()
            
        result = LearnerDecisionVar(
            timing_eps=float(arr_sum[0]),
            timing_eps_decay=float(arr_sum[1]),
            freq_eps=float(arr_sum[2]),
            freq_eps_decay=float(arr_sum[3]),
            tail_length=int(arr_sum[4])
        )
        return result

    def __sub__(self, other):
        if not isinstance(other, LearnerDecisionVar):
            return NotImplemented
        
        arr_diff = self.to_array() - other.to_array()
        
        result = LearnerDecisionVar(
            timing_eps=float(arr_diff[0]),
            timing_eps_decay=float(arr_diff[1]),
            freq_eps=float(arr_diff[2]),
            freq_eps_decay=float(arr_diff[3]),
            tail_length=int(arr_diff[4])
        )
        return result
    ######
    
    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, LearnerDecisionVar):
            return self.timing_eps == other.timing_eps and \
                   self.timing_eps_decay == other.timing_eps_decay and \
                   self.freq_eps == other.freq_eps and \
                   self.freq_eps_decay == other.freq_eps_decay and \
                   self.tail_length == other.tail_length
        return False

    # Instead of __array__; to avoid implicit conversion to numpy array
    def to_array(self, dtype=None):
        vals = [
            self.timing_eps,
            self.timing_eps_decay,
            self.freq_eps,
            self.freq_eps_decay,
            self.tail_length
        ]
        return np.array(vals, dtype=dtype)
    
    def clip(self):
        """Clip the values of the DecisionVar instance to its bounds."""
        self.timing_eps = np.clip(self.timing_eps, LearnerDecisionVar.lb[0], LearnerDecisionVar.ub[0])
        self.timing_eps_decay = np.clip(self.timing_eps_decay, LearnerDecisionVar.lb[1], LearnerDecisionVar.ub[1])
        self.freq_eps = np.clip(self.freq_eps, LearnerDecisionVar.lb[2], LearnerDecisionVar.ub[2])
        self.freq_eps_decay = np.clip(self.freq_eps_decay, LearnerDecisionVar.lb[3], LearnerDecisionVar.ub[3])
        self.tail_length = int(np.clip(self.tail_length, LearnerDecisionVar.lb[4], LearnerDecisionVar.ub[4]))

    def find_out_of_bound_dims(self):
        # Return an array of 0/1 flags indicating which fields are out of bound
        orig_vals = self.to_array(dtype=object)
        clipped_vals = np.array([
            np.clip(self.timing_eps, LearnerDecisionVar.lb[0], LearnerDecisionVar.ub[0]),
            np.clip(self.timing_eps_decay, LearnerDecisionVar.lb[1], LearnerDecisionVar.ub[1]),
            np.clip(self.freq_eps, LearnerDecisionVar.lb[2], LearnerDecisionVar.ub[2]),
            np.clip(self.freq_eps_decay, LearnerDecisionVar.lb[3], LearnerDecisionVar.ub[3]),
            int(np.clip(self.tail_length, LearnerDecisionVar.lb[4], LearnerDecisionVar.ub[4]))
        ], dtype=object)

        # 1 if original value differed from clipped value (i.e., out of bound), else 0
        out_of_bound = (orig_vals != clipped_vals).astype(bool)
        return out_of_bound

    def handle_boundary_pBest_dim(self, pbest_arr):
        # Get boolean mask of out-of-bound dimensions
        out_of_bound = self.find_out_of_bound_dims()

        # Convert pbest_arr to a numeric array if needed
        if isinstance(pbest_arr, LearnerDecisionVar):
            pbest_vals = pbest_arr.to_array(dtype=object)
        else:
            pbest_vals = np.asarray(pbest_arr, dtype=object)

        if pbest_vals.size != LearnerDecisionVar.n_dim:
            raise ValueError(f"pbest_arr must have {LearnerDecisionVar.n_dim} elements")
            
        # Use vectorized replace via to_array/from_array
        self_vals = self.to_array()
        self_vals[out_of_bound] = pbest_vals[out_of_bound]

        new = LearnerDecisionVar.from_array(self_vals)
    
        # Assign all fields from the reconstructed instance
        # print("before applying pbest handling: ", self)
        self.timing_eps = new.timing_eps
        self.timing_eps_decay = new.timing_eps_decay
        self.freq_eps = new.freq_eps
        self.freq_eps_decay = new.freq_eps_decay
        self.tail_length = new.tail_length
        # print("after applying pbest handling: ", self)
        return out_of_bound
    

class InnerLoopDecisionVar:
    @staticmethod
    def random(bin_n_dim, bin_mapping=None):
        bin_decision_var = BinaryDecisionVar.random(bin_n_dim, mapping=bin_mapping)
        learner_decision_var = LearnerDecisionVar.random()
        return InnerLoopDecisionVar(bin_decision_var, learner_decision_var)
    
    def __init__(self, bin_decision_var: BinaryDecisionVar, learner_decision_var: LearnerDecisionVar):
        self.bin_decision_var = bin_decision_var
        self.learner_decision_var = learner_decision_var
        
    def __repr__(self):
        return (f"InnerLoopDecisionVar(\n  bin_decision_var={self.bin_decision_var},\n"
                f"  learner_decision_var={self.learner_decision_var}\n)")
        
    @classmethod
    def from_array(cls, bin_arr, learner_arr, bin_mapping=None):
        bin_decision_var = BinaryDecisionVar.from_array(bin_arr, mapping=bin_mapping)
        learner_decision_var = LearnerDecisionVar.from_array(learner_arr)
        return cls(bin_decision_var, learner_decision_var)
    
    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, InnerLoopDecisionVar):
            return self.bin_decision_var == other.bin_decision_var and \
                   self.learner_decision_var == other.learner_decision_var
        return False
    
    def to_array(self, dtype=None):
        bin_array = self.bin_decision_var.arr
        learner_array = self.learner_decision_var.to_array(dtype=dtype)
        return (bin_array, learner_array) # np.concatenate((bin_array, learner_array))
        
    def handle_boundary_pBest_dim(self, pbest_learner_arr):
        return self.learner_decision_var.handle_boundary_pBest_dim(pbest_learner_arr)
        
        
######   ######   ######   ######  ######   ######   ######

class KernelType(Enum):
    # LINEAR = 0
    RBF = 0       # comment out the other two and change RBF to 0
    # COSINE = 2

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
    # lb = np.array([100, 10.0, 0, 0.05])
    # ub = np.array([1000, 500.0, len(KernelType) - 1, 10.0])
    lb = np.array([5, 5.0, 0, 0.0001])
    ub = np.array([5000, 1000.0, len(KernelType) - 1, 100.0])
        
    # in update_X and update_V, handle rbf_gamma when kernel is not rbf,
    # in here, handle discrete arithmetic
    @staticmethod
    def random():
        min_size = np.random.randint(DecisionVar.lb[0], DecisionVar.ub[0])
        penalty = np.random.uniform(DecisionVar.lb[1], DecisionVar.ub[1])
        kernel = np.random.choice(list(KernelType))
        ## Using a log-uniform distribution for rbf_gamma to cover different orders of magnitude more effectively
        rbf_gamma = np.exp(np.random.uniform(np.log(DecisionVar.lb[3]), np.log(DecisionVar.ub[3])))
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
        # self.clip()               # might be invalid! but we don't care
        
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
    
    def find_out_of_bound_dims(self):
        # Return an array of 0/1 flags indicating which fields are out of bound
        orig_vals = self.to_array(dtype=object)
        clipped_vals = np.array([
            np.clip(self.min_size, DecisionVar.lb[0], DecisionVar.ub[0]),
            np.clip(self.penalty, DecisionVar.lb[1], DecisionVar.ub[1]),
            np.clip(self.kernel.value, DecisionVar.lb[2], DecisionVar.ub[2]),
            np.clip(self.rbf_gamma, DecisionVar.lb[3], DecisionVar.ub[3])
        ], dtype=object)

        # 1 if original value differed from clipped value (i.e., out of bound), else 0
        out_of_bound = (orig_vals != clipped_vals).astype(bool)
        return out_of_bound

    def handle_boundary_pBest_dim(self, pbest_arr):
        # Get boolean mask of out-of-bound dimensions
        out_of_bound = self.find_out_of_bound_dims()

        # Convert pbest_arr to a numeric array if needed
        if isinstance(pbest_arr, DecisionVar):
            pbest_vals = pbest_arr.to_array(dtype=object)
        else:
            pbest_vals = np.asarray(pbest_arr, dtype=object)

        if pbest_vals.size != DecisionVar.n_dim:
            raise ValueError(f"pbest_arr must have {DecisionVar.n_dim} elements")
            
        # Use vectorized replace via to_array/from_array
        self_vals = self.to_array()
        self_vals[out_of_bound] = pbest_vals[out_of_bound]

        new = DecisionVar.from_array(self_vals)
    
        # Assign all fields from the reconstructed instance
        # print("before applying pbest handling: ", self)
        self.min_size = new.min_size
        self.penalty = new.penalty
        self.kernel = new.kernel
        self.rbf_gamma = new.rbf_gamma
        # print("after applying pbest handling: ", self)
        return out_of_bound

# Wrapper around multiple DecisionVar instances
class CompositeDecisionVar:
    var_names = []  # Names of the decision variables in the composite variable
    
    @staticmethod
    def set_var_names(var_names):
        CompositeDecisionVar.var_names = var_names
        
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
        vars_dict = {CompositeDecisionVar.var_names[i]: self.decision_vars[i] 
                    for i in range(self.n_vars)}
        lines = [f"  {k}: {v}" for k, v in vars_dict.items()]
        return f"CompositeDecisionVar(n_vars={self.n_vars},\n" + "\n".join(lines) + "\n)"
        # return f"CompositeDecisionVar(n_vars={self.n_vars}, decision_vars={self.decision_vars})"
    
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
    
    def handle_boundary_pBest_dim(self, pbest_learner_arr):
        out_of_bound_flags = []
        for idx, var in enumerate(self.decision_vars):
            out_of_bound_flags.append(var.handle_boundary_pBest_dim(pbest_learner_arr[idx * self.n_dim_per_var:(idx + 1) * self.n_dim_per_var]))
            self.decision_vars[idx] = var

        return np.concatenate([np.asarray(flags, dtype=bool).ravel() for flags in out_of_bound_flags]).astype(bool)

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
    