from config.settings import CONFIG

class FeatureSmoother:
    def __init__(self):
        self.smoothed = {}
        self.factors = CONFIG['smoothing']
        
    def apply(self, features):
        if not self.factors['enable']:
            return {k: round(v, 3) for k, v in features.items()}  # 保持输出精度
        
        output = {}
        for key, val in features.items():
            # 输入精度处理（接收四位小数）
            val = round(float(val), 4)
            
            # 获取平滑因子
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
                
            # 获取历史值（自动处理初始状态）
            prev = self.smoothed.get(key, round(val, 4))
            
            # 执行平滑计算
            smoothed_val = alpha * prev + (1 - alpha) * val
            
            # 输出精度处理（保留三位小数）
            output_val = round(smoothed_val, 3)
            
            # 更新存储值（保持四位精度用于下次计算）
            self.smoothed[key] = round(smoothed_val, 4)
            output[key] = output_val
            
        return output