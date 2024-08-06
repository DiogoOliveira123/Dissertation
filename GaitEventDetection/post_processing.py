import numpy as np
from scipy.signal import savgol_filter
from pykalman import KalmanFilter
import matplotlib.pyplot as plt
import os

from save_plot_true_pred import get_trial

class PostProcessingFilters:
    def __init__(self, pred_labels):
        self.pred_labels = pred_labels
        
    
    def savitzky_golay_filter(self):
        # Initial smoothing with Savitzky-Golay filter
        smoothed_predictions = savgol_filter(self.pred_labels, window_length=5, polyorder=2)
        
        return smoothed_predictions

    def kalman_filter(smoothed_predictions):
        # Further smoothing with Kalman Filter
        kf = KalmanFilter(initial_state_mean=smoothed_predictions[0], n_dim_obs=1)
        state_means, _ = kf.filter(smoothed_predictions)
        smoothed_predictions = state_means.flatten()
        return smoothed_predictions

    def IQR_filter(smoothed_predictions):
        # Optional: Remove outliers using IQR method
        Q1 = np.percentile(smoothed_predictions, 25)
        Q3 = np.percentile(smoothed_predictions, 75)
        IQR = Q3 - Q1
        filtered_predictions = smoothed_predictions[(smoothed_predictions >= Q1 - 1.5 * IQR) & (smoothed_predictions <= Q3 + 1.5 * IQR)]
        return filtered_predictions

    def snap_to_targets(filtered_predictions):
        # Snap values to nearest target (0, 1, -1)
        snapped_predictions = np.where(filtered_predictions > 0.5, 1,
                                    np.where(filtered_predictions < -0.5, -1, 0))
        return snapped_predictions

    def mode(List):
        List = [x for xs in List for x in xs]
        return int(max(set(List), key=List.count))

    def mode_filter(trial_list):
        length = len(trial_list)
        stride = 6
        # Iterate through the list with a stride of stride # 
        for i in range(0, length, stride):
            # Create a buffer of the next stride # elements (or less if at the end)
            buff = [trial_list[i:i + stride]]
            # Calculate the mode of the buffer
            corr_pred = filter.mode(buff)
            # Update the values in the original list with the mode
            trial_list[i:i + stride] = [corr_pred] * len(trial_list[i:i + stride])

        return trial_list

    def final_filter(self, true_labels, part, trial, df, base_path, data, save_plot, proc_trial):
        if proc_trial:
            filter = PostProcessingFilters()
            pred_trial, true_trial = get_trial(dataframe=df,
                                               pred_labels=self.pred_labels,
                                               true_labels=true_labels,
                                               part=part,
                                               trial=trial)

            final_prediction = filter.savitzky_golay_filter(pred_trial)
            final_prediction = filter.kalman_filter(final_prediction)
            final_prediction = filter.IQR_filter(final_prediction)
            final_prediction = filter.snap_to_targets(final_prediction)
            # final_prediction = normalize_list(final_prediction)
            final_prediction = filter.mode_filter(final_prediction)

            if save_plot:
                plt.plot(true_trial, label='True Labels')
                plt.plot(pred_trial, label='Predicted Labels')
                plt.plot(final_prediction, label='Final Predictions', linestyle='dashed')
                plt.title(f'Trial {trial+1} - Part {part} | Post-Processing')
                plt.legend()
                # plt.show()
                print('Saving plots in '+ os.path.join(base_path, 'plots_labels', data, f'Participant_{part}'))
                plt.savefig(os.path.join(base_path, 'post_processing', data, f'Participant_{part}', f'Trial{trial+1}_PostProcessing.png'))
                plt.close()

        else:
            final_prediction = filter.savitzky_golay_filter(self.pred_labels)
            final_prediction = filter.kalman_filter(final_prediction)
            final_prediction = filter.IQR_filter(final_prediction)
            final_prediction = filter.snap_to_targets(final_prediction)
            # final_prediction = normalize_list(final_prediction)
            final_prediction = filter.mode_filter(final_prediction)

        return final_prediction, pred_trial, true_trial



