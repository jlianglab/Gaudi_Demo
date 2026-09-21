import os
from sklearn.metrics import roc_auc_score
import torch
import numpy as np
import yaml
from scipy import interpolate
from PIL import Image
import matplotlib.pyplot as plt
import json 

def plot(auc_dict, cycles, dataset, nums_dataset, output_path):
    plt.figure(figsize=(10, 6))
    order = 1
    
    plt.plot(cycles, auc_dict['student'][nums_dataset-1::nums_dataset], label='Student', marker='o', markersize=3)
    plt.plot(cycles, auc_dict['teacher'][nums_dataset-1::nums_dataset], label='Teacher', marker='s', markersize=3)
    
    plt.xlabel('Cycle')
    plt.ylabel('AUC')
    plt.title('Ark %s Test AUC' % (dataset))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_path, f'{dataset}_student_teacher.png'), dpi=150)
    
    plt.close()
    
    plt.figure(figsize=(10, 6))
    
    n = nums_dataset
    
    cycles_focused = np.arange(1*(order+1)/n, len(cycles)+1/n, 1)
    focused_y = np.array(auc_dict['student'][order::n])
    
    cycles_unfocused = np.arange(1/n, len(cycles)+1/n, 1/n)
    unfocused_y = np.array(auc_dict['student'])
    
    color = 'tab:blue'
    
    # Lines only, no markers here -- markers are drawn separately below
    plt.plot(cycles_focused, focused_y, label="Focused Training", linestyle='-', color=color)
    plt.plot(cycles_unfocused, unfocused_y, label="Unfocused Training", linestyle='-', color=color, alpha=0.5)
    
    # Which unfocused indices correspond to the focused subset
    focused_mask = (np.arange(len(cycles_unfocused)) % n) == order
    
    # Points Focused Training passes through: small, closed (filled) circle
    plt.scatter(cycles_unfocused[focused_mask], unfocused_y[focused_mask],
                s=10, facecolors=color, edgecolors=color, zorder=3, alpha=1.0)
    
    # Points only Unfocused Training passes through: small, open circle
    plt.scatter(cycles_unfocused[~focused_mask], unfocused_y[~focused_mask],
                s=10, facecolors='none', edgecolors=color, zorder=3, alpha=0.5)
    
    plt.xlabel('Cycle')
    plt.ylabel('AUC')
    plt.title('Ark %s Test AUC' % (dataset))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_path, f'{dataset}_student_focus.png'), dpi=150)
    
    plt.close()

if __name__ == '__main__':
    output_path = '/scratch/echang32/Ark/Outputs/swin_base_'
    auc_file = os.path.join(output_path, 'ChestXray14_auc.json')
    with open(auc_file, 'r') as file:
       auc_dict = json.load(file) 

    cycles = range(1, 101)
    dataset = 'ChestXray14'
    num_datasets = 2
    plot(auc_dict, cycles, dataset, num_datasets, output_path)