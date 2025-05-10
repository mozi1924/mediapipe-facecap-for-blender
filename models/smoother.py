from config.settings import CONFIG

class FeatureSmoother:
    def __init__(self):
        self.smoothed = {}
        self.factors = CONFIG['smoothing']
    def apply(self, features):
        if not self.factors['enable']:
            return features.copy()
        output = {}
        for key, val in features.items():
            if 'pupil' in key:
                alpha = self.factors['pupils']
            elif key.endswith('_eyelid'):
                alpha = self.factors['eyelids']
            elif key.startswith('head'):
                alpha = self.factors['head']
            elif 'mouth' in key:
                alpha = self.factors['mouth']
            elif 'brow' in key:
                alpha = self.factors['brows']
            elif 'teeth' in key:
                alpha = self.factors['teeth']
            else:
                alpha = 0.5
            prev = self.smoothed.get(key, features[key])
            out = alpha * prev + (1-alpha) * val
            self.smoothed[key] = out
            output[key] = out
        return output