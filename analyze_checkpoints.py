#!/usr/bin/env python3
"""
Checkpoint Analysis and Recovery Script
======================================
Use this script to:
1. Check progress of sampling job
2. Extract partial results from checkpoints
3. Resume from specific checkpoint
"""

import numpy as np
import glob
import os

DATA_PATH = "/storage/homefs/mc21s017/data_vf/"
num_samples = 120

def analyze_checkpoints():
    """Analyze all available checkpoints"""
    print("=== CHECKPOINT ANALYSIS ===")
    
    # Find all checkpoint files
    checkpoint_files = glob.glob(f"{DATA_PATH}/samples_x3_checkpoint_{num_samples}_batch*.npz")
    main_checkpoint = f"{DATA_PATH}/samples_x3_checkpoint_{num_samples}.npz"
    final_checkpoint = f"{DATA_PATH}/samples_x3_FINAL_{num_samples}.npz"
    
    all_files = []
    if os.path.exists(main_checkpoint):
        all_files.append(("Main", main_checkpoint))
    
    for f in sorted(checkpoint_files):
        batch_num = f.split("batch")[1].split(".")[0]
        all_files.append((f"Batch {batch_num}", f))
    
    if os.path.exists(final_checkpoint):
        all_files.append(("FINAL", final_checkpoint))
    
    if not all_files:
        print("No checkpoint files found!")
        return None
    
    print(f"Found {len(all_files)} checkpoint files:")
    
    latest_checkpoint = None
    max_patients = 0
    
    for name, filepath in all_files:
        try:
            checkpoint = np.load(filepath)
            samples = checkpoint['samples']
            completed_batches = int(checkpoint['completed_batches'])
            completed_patients = int(checkpoint['completed_patients'])
            
            # Calculate progress
            total_patients = samples.shape[1]
            progress_pct = (completed_patients / total_patients) * 100
            
            print(f"  {name:12}: {completed_patients:4d}/{total_patients} patients "
                  f"({progress_pct:5.1f}%) - {completed_batches} batches")
            
            if completed_patients > max_patients:
                max_patients = completed_patients
                latest_checkpoint = filepath
                
        except Exception as e:
            print(f"  {name:12}: ERROR - {e}")
    
    return latest_checkpoint, max_patients

def extract_partial_results(checkpoint_file, output_suffix="partial"):
    """Extract partial results from checkpoint"""
    try:
        print(f"\nExtracting results from: {checkpoint_file}")
        checkpoint = np.load(checkpoint_file)
        samples = checkpoint['samples']
        completed_patients = int(checkpoint['completed_patients'])
        
        print(f"Full shape: {samples.shape}")
        print(f"Completed patients: {completed_patients}")
        
        # Extract only completed patients
        partial_samples = samples[:, :completed_patients, :]
        
        # Save partial results
        output_file = f"{DATA_PATH}/samples_x3_test_pred_bern_3vf_{num_samples}_{output_suffix}.npy"
        np.save(output_file, partial_samples)
        
        print(f"Saved partial results: {output_file}")
        print(f"Partial shape: {partial_samples.shape} (num_samples, num_patients, num_locations)")
        
        return output_file
        
    except Exception as e:
        print(f"Error extracting results: {e}")
        return None

def main():
    print("Checkpoint Analysis Tool")
    print("=" * 50)
    
    # Analyze checkpoints
    latest_checkpoint, max_patients = analyze_checkpoints()
    
    if latest_checkpoint:
        print(f"\nLatest checkpoint: {latest_checkpoint}")
        print(f"Max completed patients: {max_patients}")
        
        # Ask user what to do
        print("\nOptions:")
        print("1. Extract partial results from latest checkpoint")
        print("2. Extract partial results from specific checkpoint")
        print("3. Just show analysis (default)")
        
        try:
            choice = input("\nEnter choice (1-3, default=3): ").strip()
            
            if choice == "1":
                extract_partial_results(latest_checkpoint, "latest")
            elif choice == "2":
                checkpoint_file = input("Enter checkpoint file path: ").strip()
                if os.path.exists(checkpoint_file):
                    extract_partial_results(checkpoint_file, "custom")
                else:
                    print("File not found!")
            else:
                print("Analysis complete.")
                
        except KeyboardInterrupt:
            print("\nAnalysis complete.")
    
    else:
        print("No valid checkpoints found!")

if __name__ == "__main__":
    main()