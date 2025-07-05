"""
Temporal-Enhanced SE-SGformer: Main Comparison Script
====================================================
This script compares the original SE-SGformer with the temporal-enhanced version
that uses historical context from previous timesteps.

The comparison evaluates whether incorporating historical node embeddings from
previous timesteps improves link prediction performance in dynamic graph neural
networks. This is particularly relevant for evolving social networks, trust
dynamics, and temporal relationship prediction tasks.

Key Features:
- Side-by-side comparison of baseline vs temporal-enhanced models
- Comprehensive performance metrics (AUC, F1, Precision@100)
- Configurable temporal context integration strategies
- Automated visualization generation for results analysis
- Reproducible experimental setup with proper seeding

Usage:
    python main.py --file_path data/soc-sign-bitcoin.csv.gz --epochs 50 --num_time_bins 6
    python main.py --file_path data/soc-sign-bitcoin.csv.gz --adaptive_weights --base_weights 0.4
"""

import argparse
import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to path for imports
# This ensures the script can find custom modules regardless of execution directory
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from config import Args
from data_loader import load_dataset_timesteps
from model import SE_SGformer, Temporal_SE_SGformer
from utilities import (calculate_precision_at_100, plot_training_loss_curves, 
                      plot_loss_difference, plot_auc_f1_comparison, plot_precision_100_comparison,
                      plot_absolute_improvements, plot_percentage_improvements)

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================

# Automatically select the best available compute device for training
# Priority order reflects typical performance characteristics:
# 1. CUDA GPU - Fastest for large-scale deep learning
# 2. Apple Silicon MPS - Optimized for Apple's neural engine
# 3. CPU - Universal fallback, slower but always available
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
    print("Using Apple Silicon MPS")
else:
    device = torch.device('cpu')
    print("Using CPU")


def parse_arguments():
    """
    Parse and validate command line arguments for the comparison experiment.
    
    This function defines all configurable parameters for the temporal GNN
    comparison, organized into logical groups for better usability.
    
    Returns:
        argparse.Namespace: Parsed arguments with default values applied.
        
    Argument Categories:
        - Data: Dataset path, temporal binning configuration
        - Training: Optimization parameters, regularization settings
        - Model: Architecture configuration, embedding dimensions
        - Temporal: Historical context integration strategies
        - Output: Result storage, visualization options
        - Debug: Reproducibility, verbose logging
    """
    parser = argparse.ArgumentParser(
        description='Temporal-Enhanced SE-SGformer Comparison',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # ========================================================================
    # DATA ARGUMENTS
    # ========================================================================
    # These parameters control how temporal graph data is loaded and processed
    
    parser.add_argument('--file_path', type=str, required=True,
                       help='Path to the dataset file (CSV or CSV.gz)')
    parser.add_argument('--num_time_bins', type=int, default=6,
                       help='Number of timesteps to split data into')
    parser.add_argument('--target_timestep', type=int, default=-1,
                       help='Target timestep for prediction (-1 for last)')
    
    # ========================================================================
    # TRAINING ARGUMENTS
    # ========================================================================
    # Core optimization settings that affect model convergence and performance
    
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                       help='Weight decay for regularization')
    parser.add_argument('--no_scheduler', action='store_true',
                       help='Disable learning rate scheduler')
    
    # ========================================================================
    # MODEL ARGUMENTS  
    # ========================================================================
    # Architecture parameters that define model capacity and structure
    
    parser.add_argument('--num_layers', type=int, default=2,
                       help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=4,
                       help='Number of attention heads')
    parser.add_argument('--node_dim', type=int, default=None,
                       help='Node embedding dimension (auto if None)')
    parser.add_argument('--max_degree', type=int, default=None,
                       help='Maximum degree for centrality encoding (auto if None)')
    
    # ========================================================================
    # TEMPORAL ARGUMENTS
    # ========================================================================
    # Configuration for historical context integration strategies
    
    parser.add_argument('--base_weights', type=float, default=0.3,
                       help='Weight for historical context combination')
    parser.add_argument('--adaptive_weights', action='store_true',
                       help='Use adaptive weights instead of fixed weights')
    
    # ========================================================================
    # OUTPUT ARGUMENTS
    # ========================================================================
    # Control result storage, visualization, and experiment tracking
    
    parser.add_argument('--title', type=str, default=None,
                       help='Title prefix for plots')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Directory to save results and plots')
    parser.add_argument('--save_models', action='store_true',
                       help='Save trained models')
    parser.add_argument('--no_plots', action='store_true',
                       help='Skip generating plots')
    
    # ========================================================================
    # DEBUGGING ARGUMENTS
    # ========================================================================
    # Tools for reproducibility and detailed experiment monitoring
    
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    return parser.parse_args()


def set_random_seed(seed):
    """
    Set random seeds across all relevant libraries for reproducible experiments.
    
    Reproducibility is crucial for fair model comparisons and scientific validity.
    This function ensures that random initialization, data shuffling, and
    stochastic training processes produce consistent results across runs.
    
    Args:
        seed (int): Random seed value to use across all libraries.
        
    Note:
        Sets seeds for PyTorch (CPU and GPU), NumPy, and Python's random module.
        GPU operations may still have minor non-deterministic behavior due to
        hardware-level optimizations.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def create_output_directory(output_dir):
    """
    Create the output directory structure for storing experiment results.
    
    This ensures that all generated files (models, plots, metrics) have a
    designated location, preventing clutter and enabling organized result
    tracking across multiple experiments.
    
    Args:
        output_dir (str): Path to the desired output directory.
        
    Returns:
        str: The validated output directory path.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return output_dir


def validate_arguments(args):
    """
    Validate command line arguments to prevent runtime errors and invalid configurations.
    
    Early validation helps catch configuration mistakes before expensive training
    begins, saving computational resources and providing clear error messages.
    
    Args:
        args (argparse.Namespace): Parsed command line arguments.
        
    Raises:
        FileNotFoundError: If the specified dataset file doesn't exist.
        ValueError: If numeric parameters are outside valid ranges.
    """
    # Verify dataset file exists to prevent training failures
    if not os.path.exists(args.file_path):
        raise FileNotFoundError(f"Dataset file not found: {args.file_path}")
    
    # Ensure minimum temporal bins for meaningful temporal analysis
    if args.num_time_bins < 2:
        raise ValueError("num_time_bins must be at least 2")
    
    # Validate training configuration
    if args.epochs < 1:
        raise ValueError("epochs must be at least 1")
    
    # Ensure weight parameters are in valid probability range
    if not (0.0 <= args.base_weights <= 1.0):
        raise ValueError("base_weights must be between 0.0 and 1.0")


def compare_approaches(file_path: str, target_timestep: int = -1, num_time_bins: int = 6, 
                      epochs: int = 50, lr: float = 0.001, weight_decay: float = 5e-4,
                      use_scheduler: bool = True, base_weights: float = 0.3, 
                      use_adaptive_weights: bool = False, title: str = None,
                      output_dir: str = 'results', save_models: bool = False,
                      generate_plots: bool = True, verbose: bool = False,
                      **model_kwargs):
    """
    Execute comprehensive comparison between baseline and temporal-enhanced SE-SGformer models.
    
    This is the core experimental function that trains both model variants under
    identical conditions and evaluates their performance across multiple metrics.
    The comparison provides insights into the value of historical context for
    temporal graph neural networks.
    
    Args:
        file_path (str): Path to the temporal graph dataset.
        target_timestep (int): Timestep to predict (-1 for last available).
        num_time_bins (int): Number of temporal segments to create.
        epochs (int): Training epochs for both models.
        lr (float): Learning rate for optimization.
        weight_decay (float): L2 regularization strength.
        use_scheduler (bool): Whether to use adaptive learning rate scheduling.
        base_weights (float): Weight for combining historical and current embeddings.
        use_adaptive_weights (bool): Use learnable vs fixed combination weights.
        title (str): Prefix for generated plot titles.
        output_dir (str): Directory for saving results and visualizations.
        save_models (bool): Whether to persist trained model parameters.
        generate_plots (bool): Whether to create performance visualizations.
        verbose (bool): Enable detailed progress logging.
        **model_kwargs: Additional model architecture parameters.
        
    Returns:
        dict: Comprehensive results including performance metrics, improvements,
              and experimental configuration for reproducibility.
              
    The function follows a structured experimental protocol:
    1. Data Loading: Load and validate temporal graph data
    2. Configuration: Set up model parameters with automatic sizing
    3. Baseline Training: Train standard SE-SGformer without temporal context
    4. Temporal Training: Train enhanced model with historical embeddings
    5. Evaluation: Compare performance across multiple metrics
    6. Visualization: Generate comprehensive result plots
    7. Result Storage: Save metrics and configuration for analysis
    """
    
    if verbose:
        print("="*70)
        print("SE-SGFORMER: HISTORICAL CONTEXT vs BASELINE COMPARISON")
        print("="*70)
    
    # ========================================================================
    # PHASE 1: DATA LOADING AND VALIDATION
    # ========================================================================
    # Load temporal graph data and verify it meets experimental requirements
    
    try:
        timesteps, num_nodes = load_dataset_timesteps(file_path, num_time_bins)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None
    
    # Ensure sufficient temporal data for meaningful comparison
    if len(timesteps) < 2:
        print("Need at least 2 timesteps for temporal comparison")
        return None
    
    # Configure target prediction timestep
    # Default (-1) uses the most recent timestep as prediction target
    if target_timestep == -1:
        target_timestep = len(timesteps) - 1
    
    if target_timestep >= len(timesteps):
        print(f"Target timestep {target_timestep} exceeds available timesteps {len(timesteps)}")
        return None
    
    # Separate target data from historical context
    # Target: the timestep we want to predict
    # Historical: all previous timesteps used for temporal context
    target_data = timesteps[target_timestep]
    historical_data = timesteps[:target_timestep] if target_timestep > 0 else []
    
    if verbose:
        print(f"\n Experiment Setup:")
        print(f"   Target timestep: {target_data['timestep']} ({target_data['num_edges']} edges)")
        print(f"   Historical timesteps: {len(historical_data)}")
        print(f"   Total nodes: {num_nodes}")
        print(f"   Epochs: {epochs}")
    
    # ========================================================================
    # PHASE 2: MODEL CONFIGURATION
    # ========================================================================
    # Configure model architecture with automatic parameter sizing based on graph scale
    
    # Adaptive parameter sizing based on graph characteristics
    # Larger graphs need higher capacity, but we avoid excessive parameters
    model_config = {
        # Degree encoding capacity scales with graph connectivity
        'max_degree': min(50, max(20, int(np.sqrt(num_nodes)))),
        # Feature dimensions scale with node count but remain manageable
        'num_node_features': min(128, max(64, num_nodes // 10)),
        'node_dim': min(128, max(64, num_nodes // 10)),
        'output_dim': min(128, max(64, num_nodes // 10)),
        # Temporal integration parameters
        'base_weights': base_weights,
        'use_adaptive_weights': use_adaptive_weights
    }
    
    # Override automatic sizing with user-specified values
    model_config.update({k: v for k, v in model_kwargs.items() if v is not None})
    
    # Create configuration object for model initialization
    args = Args(**model_config)
    
    if verbose:
        print(f"   Model dimensions: {args.node_dim}")
        print(f"   Use adaptive weights: {use_adaptive_weights}")
    
    # Initialize training loss tracking for both models
    baseline_losses = []
    temporal_losses = []
    
    # ========================================================================
    # PHASE 3: BASELINE MODEL TRAINING
    # ========================================================================
    # Train standard SE-SGformer without temporal context as performance baseline
    
    if verbose:
        print(f"\n Training Baseline Model...")
        print("-" * 50)
    
    try:
        # Initialize baseline model and move to appropriate device
        baseline_model = SE_SGformer(args).to(device)
        
        # Create spectral node features for the target timestep
        # These features capture graph structure but not temporal patterns
        x_target = baseline_model.create_spectral_features(
            target_data['pos_edge_index'], target_data['neg_edge_index'], num_nodes
        )
        
        # Configure optimizer with weight decay for regularization
        optimizer_baseline = torch.optim.Adam(baseline_model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Optional learning rate scheduling for better convergence
        if use_scheduler:
            scheduler_baseline = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer_baseline, mode='min', factor=0.8, patience=20, min_lr=1e-6)
        
        # Training loop for baseline model
        baseline_model.train()
        for epoch in range(epochs):
            optimizer_baseline.zero_grad()
            
            # Forward pass: compute node embeddings
            z = baseline_model(x_target, target_data['pos_edge_index'], target_data['neg_edge_index'])
            
            # Compute link prediction loss
            loss = baseline_model.loss(z, target_data['pos_edge_index'], target_data['neg_edge_index'])
            
            # Backward pass and optimization
            loss.backward()
            
            # Gradient clipping prevents training instability
            torch.nn.utils.clip_grad_norm_(baseline_model.parameters(), max_norm=1.0)
            optimizer_baseline.step()
            
            # Update learning rate based on loss plateau
            if use_scheduler:
                scheduler_baseline.step(loss.item())
            
            # Track training progress
            baseline_losses.append(loss.item())
            
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"   Epoch {epoch}: Loss = {loss.item():.4f}")
        
        # ====================================================================
        # BASELINE MODEL EVALUATION
        # ====================================================================
        # Evaluate baseline performance across multiple metrics
        
        baseline_model.eval()
        with torch.no_grad():
            # Generate final embeddings for evaluation
            z_baseline = baseline_model(x_target, target_data['pos_edge_index'], target_data['neg_edge_index'])
            
            # Compute standard link prediction metrics
            baseline_auc, baseline_f1 = baseline_model.test(z_baseline, target_data['pos_edge_index'], target_data['neg_edge_index'])
            
            # Compute precision at top-100 predictions (ranking-based metric)
            baseline_precision_100 = calculate_precision_at_100(
                baseline_model, x_target, target_data['pos_edge_index'], target_data['neg_edge_index']
            )
        
        # Save baseline model if requested
        if save_models:
            torch.save(baseline_model.state_dict(), f"{output_dir}/baseline_model.pt")
        
        print(f"Baseline Results: AUC = {baseline_auc:.4f}, F1 = {baseline_f1:.4f}, P@100 = {baseline_precision_100:.4f}")
        
    except Exception as e:
        print(f"Error training baseline model: {e}")
        return None
    
    # ========================================================================
    # PHASE 4: TEMPORAL MODEL TRAINING
    # ========================================================================
    # Train temporal-enhanced SE-SGformer with historical context integration
    
    if verbose:
        print(f"\n Training Temporal Model...")
        print("-" * 50)
    
    try:
        # Initialize temporal-enhanced model
        temporal_model = Temporal_SE_SGformer(args).to(device)
        
        # ====================================================================
        # HISTORICAL CONTEXT EXTRACTION
        # ====================================================================
        # Extract embeddings from previous timesteps to provide temporal context
        
        historical_embeddings = []
        if historical_data:
            if verbose:
                print(f"   Extracting context from {len(historical_data)} previous timesteps...")
            
            # Process each historical timestep to extract node embeddings
            with torch.no_grad():
                for hist_data in historical_data:
                    # Create features for historical timestep
                    x_hist = temporal_model.base_model.create_spectral_features(
                        hist_data['pos_edge_index'], hist_data['neg_edge_index'], num_nodes
                    )
                    
                    # Generate embeddings that capture graph state at this timestep
                    z_hist = temporal_model.base_model(x_hist, hist_data['pos_edge_index'], hist_data['neg_edge_index'])
                    historical_embeddings.append(z_hist)
        
        # Configure optimizer for temporal model
        optimizer_temporal = torch.optim.Adam(temporal_model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Learning rate scheduling for temporal model
        if use_scheduler:
            scheduler_temporal = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer_temporal, mode='min', factor=0.8, patience=20, min_lr=1e-6)
        
        # Training loop for temporal model
        temporal_model.train()
        for epoch in range(epochs):
            optimizer_temporal.zero_grad()
            
            # Forward pass with historical context
            # This is where temporal enhancement happens - the model combines
            # current graph features with historical embeddings
            z = temporal_model(x_target, target_data['pos_edge_index'], target_data['neg_edge_index'], historical_embeddings)
            
            # Compute loss (same objective as baseline for fair comparison)
            loss = temporal_model.loss(z, target_data['pos_edge_index'], target_data['neg_edge_index'])
            
            # Standard optimization steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(temporal_model.parameters(), max_norm=1.0)
            optimizer_temporal.step()
            
            if use_scheduler:
                scheduler_temporal.step(loss.item())
            
            temporal_losses.append(loss.item())
            
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"   Epoch {epoch}: Loss = {loss.item():.4f}")
        
        # ====================================================================
        # TEMPORAL MODEL EVALUATION
        # ====================================================================
        # Evaluate temporal model performance using same metrics as baseline
        
        temporal_model.eval()
        with torch.no_grad():
            # Generate embeddings with temporal context
            z_temporal = temporal_model(x_target, target_data['pos_edge_index'], target_data['neg_edge_index'], historical_embeddings)
            
            # Compute performance metrics
            temporal_auc, temporal_f1 = temporal_model.test(z_temporal, target_data['pos_edge_index'], target_data['neg_edge_index'])
            
            # Precision@100 with temporal context
            temporal_precision_100 = calculate_precision_at_100(
                temporal_model, x_target, target_data['pos_edge_index'], target_data['neg_edge_index'], 
                historical_embeddings=historical_embeddings
            )
        
        # Save temporal model if requested
        if save_models:
            torch.save(temporal_model.state_dict(), f"{output_dir}/temporal_model.pt")
        
        print(f"Temporal Results: AUC = {temporal_auc:.4f}, F1 = {temporal_f1:.4f}, P@100 = {temporal_precision_100:.4f}")
        
    except Exception as e:
        print(f"Error training temporal model: {e}")
        return None
    
    # ========================================================================
    # PHASE 5: PERFORMANCE COMPARISON AND ANALYSIS
    # ========================================================================
    # Compute improvement metrics and statistical significance of temporal enhancement
    
    print(f"\n FINAL COMPARISON")
    print("=" * 50)
    
    # Calculate absolute improvements
    auc_improvement = temporal_auc - baseline_auc
    f1_improvement = temporal_f1 - baseline_f1
    precision_improvement = temporal_precision_100 - baseline_precision_100
    
    # Calculate percentage improvements (relative to baseline performance)
    auc_improvement_pct = (auc_improvement / baseline_auc * 100) if baseline_auc > 0 else 0
    f1_improvement_pct = (f1_improvement / baseline_f1 * 100) if baseline_f1 > 0 else 0
    precision_improvement_pct = (precision_improvement / baseline_precision_100 * 100) if baseline_precision_100 > 0 else 0
    
    # Display comprehensive comparison results
    print(f"AUC:       {baseline_auc:.4f} → {temporal_auc:.4f} ({auc_improvement:+.4f}, {auc_improvement_pct:+.1f}%)")
    print(f"F1:        {baseline_f1:.4f} → {temporal_f1:.4f} ({f1_improvement:+.4f}, {f1_improvement_pct:+.1f}%)")
    print(f"P@100:     {baseline_precision_100:.4f} → {temporal_precision_100:.4f} ({precision_improvement:+.4f}, {precision_improvement_pct:+.1f}%)")
    
    # ========================================================================
    # PHASE 6: VISUALIZATION GENERATION
    # ========================================================================
    # Create comprehensive visualizations for result analysis and presentation
    
    if generate_plots:
        print(f"\n Generating Visualizations...")
        
        # Prepare results in standard format for plotting functions
        baseline_results = {
            'auc': baseline_auc,
            'f1': baseline_f1,
            'precision_at_100': baseline_precision_100
        }
        
        temporal_results = {
            'auc': temporal_auc,
            'f1': temporal_f1,
            'precision_at_100': temporal_precision_100
        }
        
        try:
            # Generate comprehensive visualization suite
            # Each plot provides different insights into model performance
            
            # Training dynamics comparison
            plot_training_loss_curves(baseline_losses, temporal_losses, title=title)
            
            # Loss difference analysis (convergence patterns)
            plot_loss_difference(baseline_losses, temporal_losses, title=title)
            
            # Performance metric comparisons
            plot_auc_f1_comparison(baseline_results, temporal_results, title=title)
            plot_precision_100_comparison(baseline_results, temporal_results, title=title)

            # Improvement analysis visualizations
            plot_absolute_improvements(baseline_results, temporal_results, title=title)
            plot_percentage_improvements(baseline_results, temporal_results, title=title)
            
        except Exception as e:
            print(f"Warning: Error generating plots: {e}")
    
    # ========================================================================
    # PHASE 7: RESULTS COMPILATION AND STORAGE
    # ========================================================================
    # Compile comprehensive results dictionary for analysis and reproducibility
    
    # Compile all experimental results into structured format
    results = {
        # Experimental setup information
        'target_timestep': target_data['timestep'],
        'num_historical': len(historical_data),
        'num_nodes': num_nodes,
        
        # Baseline model performance
        'baseline': {
            'auc': baseline_auc,
            'f1': baseline_f1,
            'precision_at_100': baseline_precision_100
        },
        
        # Temporal model performance
        'temporal': {
            'auc': temporal_auc,
            'f1': temporal_f1,
            'precision_at_100': temporal_precision_100
        },
        
        # Improvement analysis
        'improvement': {
            'auc': auc_improvement,
            'f1': f1_improvement,
            'precision_100': precision_improvement,
            'auc_pct': auc_improvement_pct,
            'f1_pct': f1_improvement_pct,
            'precision_100_pct': precision_improvement_pct
        },
        
        # Configuration for reproducibility
        'config': vars(args),
    }
    
    # Save results to JSON file for later analysis
    if output_dir:
        results_file = f"{output_dir}/comparison_results.json"
        try:
            import json
            with open(results_file, 'w') as f:
                # Convert tensors to floats for JSON serialization
                json_results = {k: float(v) if torch.is_tensor(v) else v for k, v in results.items() if k != 'config'}
                json.dump(json_results, f, indent=2)
            if verbose:
                print(f"Results saved to {results_file}")
        except Exception as e:
            if verbose:
                print(f"Warning: Could not save results: {e}")
    
    return results


def main():
    """
    Main entry point for the temporal GNN comparison experiment.
    
    This function orchestrates the complete experimental pipeline:
    1. Parse and validate command line arguments
    2. Configure reproducible experimental environment  
    3. Execute the comparison between baseline and temporal models
    4. Handle errors gracefully and provide informative feedback
    
    The function implements proper error handling and user feedback to ensure
    robust execution across different environments and configurations.
    """
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Set random seed for reproducible experiments
        # This is crucial for fair model comparison
        set_random_seed(args.seed)
        
        # Validate arguments before expensive computations
        validate_arguments(args)
        
        # Ensure output directory exists
        create_output_directory(args.output_dir)
        
        # Display configuration for experiment tracking
        if args.verbose:
            print("Configuration:")
            for key, value in vars(args).items():
                print(f"   {key}: {value}")
            print()
        
        # Execute the main comparison experiment
        results = compare_approaches(
            # Data configuration
            file_path=args.file_path,
            target_timestep=args.target_timestep,
            num_time_bins=args.num_time_bins,
            
            # Training configuration
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            use_scheduler=not args.no_scheduler,
            
            # Temporal configuration
            base_weights=args.base_weights,
            use_adaptive_weights=args.adaptive_weights,
            
            # Output configuration
            title=args.title,
            output_dir=args.output_dir,
            save_models=args.save_models,
            generate_plots=not args.no_plots,
            verbose=args.verbose,
            
            # Model architecture parameters
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            node_dim=args.node_dim,
            max_degree=args.max_degree
        )
        
        # Handle experiment failure
        if results is None:
            print("Comparison failed")
            sys.exit(1)
        
        # Display experiment summary
        print(f"\n Summary:")
        print(f"   Best improvement: {max(results['improvement']['auc_pct'], results['improvement']['f1_pct'], results['improvement']['precision_100_pct']):.1f}%")
        
        if args.verbose:
            print(f"   Results available in: {args.output_dir}")
        
    except KeyboardInterrupt:
        # Handle user interruption gracefully
        print("\n Interrupted by user")
        sys.exit(1)
    except Exception as e:
        # Handle unexpected errors with appropriate detail level
        print(f"Error: {e}")
        if args.verbose if 'args' in locals() else False:
            import traceback
            traceback.print_exc()
        sys.exit(1)


# Execute main function when script is run directly
if __name__ == "__main__":
    main()