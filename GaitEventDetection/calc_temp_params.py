import numpy as np


class TemporalParameters:
    def __init__(self, freq):
        self.freq = freq

    def get_step_time(self, hs1, hs2):
        step_time = []
        for x1, x2 in zip(hs1, hs2):
            calc_step_time = 1/self.freq * (x1 - x2)
            step_time.append(round(calc_step_time, 2))

        return step_time

    def get_stride_time(self, hs):
        hs = np.array(hs)
        stride_time = [round((1 / self.freq) * x, 2) for x in list(np.diff(hs))]

        return stride_time

    def get_stance_time(self, to, hs):
        stance_time = []
        for t, h in zip(to, hs):
            calc_stance_time = 1 / self.freq * (t - h)
            stance_time.append(round(calc_stance_time, 2))

        return stance_time

    def get_swing_time(self, hs, to):
        swing_time = []
        for h, t in zip(hs, to):
            calc_swing_time = 1 / self.freq * (h - t)
            swing_time.append(round(calc_swing_time, 2))

        return swing_time

    def get_asymm_index(self, p_right, p_left):
        # Gait Asymmetry Index (GAI)
        gai = (2 * abs(p_left - p_right) / (p_left + p_right)) * 100

        return gai