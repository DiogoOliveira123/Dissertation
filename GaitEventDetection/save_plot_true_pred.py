import os
import matplotlib.pyplot as plt

def get_trial(dataframe, pred_labels, true_labels, part, trial):
    part_trial = f'Participant_{part}\\Trial_{trial+1}' 
    indexes = [
        idx for idx in range(dataframe.shape[0]) 
        if part_trial in dataframe.iloc[idx, 0] and 
        (dataframe.iloc[idx, 0].split(part_trial)[-1] == '' or 
        not dataframe.iloc[idx, 0].split(part_trial)[-1][0].isdigit()
        )
        ]
    
    first_idx = indexes[0]
    last_idx = indexes[-1]
    pred_trial = pred_labels[first_idx:last_idx]
    true_trial = true_labels[first_idx:last_idx]

    for i in range(len(pred_trial)):
        if pred_trial[i] == 1:
            pred_trial[i] = 0
        elif pred_trial[i] == 0:
            pred_trial[i] = -1
        else:
            pred_trial[i] = 1

    for i in range(len(true_trial)):
        if true_trial[i] == 1:
            true_trial[i] = 0
        elif true_trial[i] == 0:
            true_trial[i] = -1
        else:
            true_trial[i] = 1
    
    return pred_trial, true_trial

# Get TRIALS indexes
def save_plot_true_pred(participants, generator, data, pred_labels, true_labels, base_path):
    NUM_TRIALS = 24
    df = generator.dataframe
    for part in participants:
        for trial in range(NUM_TRIALS):
            pred_trial, true_trial = get_trial(dataframe=df,
                                               pred_labels=pred_labels,
                                               true_labels=true_labels,
                                               part=part,
                                               trial=trial)

            # Plot true labels vs model's predicted labels
            y_range = len(pred_trial)
            y_axis = [x for x in range(y_range)]
            plt.plot(y_axis, true_trial, label='True')
            plt.plot(y_axis, pred_trial, label='Predict')
            plt.title(f'Trial {trial+1} - Part {part} | True vs Predicted Labels')
            # plt.show()
            print('Saving plots in '+ os.path.join(base_path, 'plots_labels', data, f'Participant_{part}'))
            plt.savefig(os.path.join(base_path, 'plots_labels', data, f'Participant_{part}', f'Trial{trial+1}_TruePred_labels.png'))
            plt.close()


def get_TP_rate(participants, generator, data, pred_labels, true_labels, base_path):
    NUM_TRIALS = 24
    df = generator.dataframe
    
    for part in participants:
        for trial in range(NUM_TRIALS):
            pred_trial, true_trial = get_trial(dataframe=df,
                                               pred_labels=pred_labels,
                                               true_labels=true_labels,
                                               part=part,
                                               trial=trial)
            
            num_TP = 0
            trial_size = len(pred_trial)
            for i in range(trial_size):
                if true_trial[i] == pred_trial[i]:
                    num_TP += 1
            
            path = os.path.join(base_path, 'TP_rate', data, f'Part{part}_TP_rates.txt')
            with open(path, 'a') as f:
                f.write(f'Trial {trial+1}:\n')
                f.write(f'Number of TP (True Positives): {num_TP}\n')
                f.write(f'Number of labels: {trial_size}\n')
                f.write(f'Rate of TP: {round(num_TP / trial_size, 2)}\n\n')


            

