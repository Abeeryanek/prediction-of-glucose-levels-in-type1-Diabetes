
import clarke_error_grid as ceg
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
def clarke_grid_import(results,model):
    for label in ['15min', '30min', '45min']:
        y_true = results[label][f'{model}_full']['y_te'].values
    
        # Full features
        y_pred_full = results[label][f'{model}_full']['preds']

        fig= ceg.plot(y_true, y_pred_full)
        fig.update_layout(title_text=f'Clarke Error Grid — {model} Full — {label.upper()}')
        print(f"--- {label.upper()} {model} FULL ZONES ---")
        zones = ceg.zone(y_true, y_pred_full)
        print(zones)
        fig.write_html(f'clarke_{model}_full_{label}.html')
        
        # Comparable
        y_true_compare = results[label][f'{model}_comparable']['y_te'].values
        y_pred_compare = results[label][f'{model}_comparable']['preds']
        print(f"--- {label.upper()} {model} COMPARABLE ZONES ---")
        zones_compare = ceg.zone(y_true_compare, y_pred_compare)
        print(zones_compare)
        fig = ceg.plot(y_true_compare, y_pred_compare)
        fig.update_layout(title_text=f'Clarke Error Grid — {model} Comparable — {label.upper()}')
        fig.write_html(f'clarke_{model}_comparable_{label}.html')
#optional
def create_bar_plot(results):
    df_results = pd.DataFrame(results).T 
    # Plot directly from the DataFrame
    fig, ax = plt.subplots(figsize=(12, 8), layout='constrained')
    df_results.plot(kind='bar', ax=ax, width=0.7, edgecolor='grey')

    # Add the value labels to the top of the bars
    for name in ax.containers:
        ax.bar_label(name, fmt='%.2f', padding=3)

    #Clean up labels
    ax.set_ylabel('RMSE (mg/dL)', fontweight='bold')
    ax.set_title('Ablation Study: RMSE per Feature Group', fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    #saved file desitnation not written!
    plt.savefig() 