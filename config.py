class Args:
    """
    Configuration container for SE-SGformer model.
    
    The sole purpose of this class is to store hyper‑parameters in a single
    discoverable place so all elements of the code base can reference the
    same source of truth.
    """
    def __init__(self, **kwargs):
        self.num_layers = 2          # Transformer blocks in the encoder stack
        self.num = 4                 # Generic size parameter (see docstring)
        self.num_node_features = 128 # Dimensional raw input per node
        self.node_dim = 128          # Hidden/attention embedding dimension
        self.output_dim = 128        # Readout / projection dimension
        self.num_heads = 4           # Parallel attention heads
        self.max_degree = 20         # Cut‑off for degree embeddings
        self.length = 50             # Temporal window length
        self.max_hop = 7             # Structural neighbourhood radius (k‑hop)
        self.use_adaptive_weights = False
        self.base_weights = 0.3      # Static edge‑weight scaling factor
        
        # Override defaults with any user‑supplied keyword arguments.
        # Using setattr keeps potential property setters intact
        for key, value in kwargs.items():
            setattr(self, key, value)