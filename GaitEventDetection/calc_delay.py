from sklearn.metrics import confusion_matrix

def get_events(labels):
    """
    Get events' indexes from a given trial.
    :param labels: list of a trial's labels
    :return: list with all the events' indexes 
    """
    events = []
    
    for i in range(1, len(labels) - 1):
        if labels[i - 1] == 0 and labels[i] == -1:
            events.append(i)
        elif labels[i - 1] == 0 and labels[i] == 1:
            events.append(i)
        elif labels[i - 1] == -1 and labels[i] == 0:
            events.append(i)
        elif labels[i - 1] == 1 and labels[i] == 0:
            events.append(i)

    return events


def get_delay(true_labels, filtered_labels):
    """
    Calculate the delay between events of different labeling types.
    :param true_labels: list of ground-truth labels
    :param filtered_labels: list of filtered labels
    :return: list of events' delays
    """
    t_events = get_events(true_labels)
    f_events = get_events(filtered_labels)
    delay_list = []

    for i in range(len(t_events) - 1):
        if i == len(t_events) or  i == len(f_events):
            break

        else:
            delay = abs(f_events[i] - t_events[i])
            delay_list.append(delay)
        
    return delay_list

    
    