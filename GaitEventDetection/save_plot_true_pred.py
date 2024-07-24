import os
import matplotlib.pyplot as plt


# Get TRIALS indexes
def save_plot_true_pred(participants, generator, data, pred_labels, true_labels, base_path):
    NUM_TRIALS = 24
    for part in participants:
        df = generator.dataframe
        for trial in range(NUM_TRIALS):
            idx = 0
            part_trial = f'Participant_{part}\\Trial_{trial+1}' 
            indexes = [
                idx for idx in range(df.shape[0]) 
                if part_trial in df.iloc[idx, 0] and 
                (df.iloc[idx, 0].split(part_trial)[-1] == '' or 
                not df.iloc[idx, 0].split(part_trial)[-1][0].isdigit()
                )
                ]
            
            first_idx = indexes[0]
            last_idx = indexes[-1]
            val_pred_trial = pred_labels[first_idx:last_idx]
            val_true_trial = true_labels[first_idx:last_idx]

            for i in range(len(val_pred_trial)):
                if val_pred_trial[i] == 1:
                    val_pred_trial[i] = 0
                elif val_pred_trial[i] == 0:
                    val_pred_trial[i] = -1
                else:
                    val_pred_trial[i] = 1

            for i in range(len(val_true_trial)):
                if val_true_trial[i] == 1:
                    val_true_trial[i] = 0
                elif val_true_trial[i] == 0:
                    val_true_trial[i] = -1
                else:
                    val_true_trial[i] = 1

            y_range = len(pred_labels[first_idx:last_idx])
            # Plot true labels vs model's predicted labels
            y_axis = [x for x in range(y_range)]
            plt.plot(y_axis, val_true_trial, label='True')
            plt.plot(y_axis, val_pred_trial, label='Predict')
            plt.title(f'Trial {trial+1} - Part {part} | True vs Predicted Labels')
            # plt.show()
            print('Saving plots in '+ os.path.join(base_path, 'plots_labels', data, f'Participant_{part}'))
            plt.savefig(os.path.join(base_path, 'plots_labels', data, f'Participant_{part}', f'Trial{trial+1}_TruePred_labels.png'))
            plt.close()