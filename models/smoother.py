from config.settings import CONFIG

class FeatureSmoother:
    def __init__(self):
        self.smoothed = {}
        self.factors = CONFIG['smoothing']
        
    def apply(self, features):
        if not self.factors['enable']:
            return features  # 直接返回原始值
            
        output = {}
        for key, val in features.items():
            # 移除输入精度处理，直接转为浮点数
            val = float(val)
            
            # 获取平滑因子（保持原有逻辑不变）
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
                
            # 获取历史值（使用原始精度）
            prev = self.smoothed.get(key, val)
            
            # 执行平滑计算
            smoothed_val = alpha * prev + (1 - alpha) * val
            
            # 更新存储值和输出值（保持原始精度）
            self.smoothed[key] = smoothed_val
            output[key] = smoothed_val
            
        return output