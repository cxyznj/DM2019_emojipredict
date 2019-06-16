import os
import numpy as np



def genfeature(features, dataidx, feature_len, filename):
    if os.path.exists(filename):
        print("载入已处理特征...")
        return np.load(filename)
    else:
        print("处理特征中...")
        result = []
        # idxes为一句话的全部特征下标
        for idxes in dataidx:
            # raw_result为二维张量，表示原始特征
            raw_feature = []
            for idx in idxes:
                raw_feature.append(features[idx])
            # k为一句话中的词数
            k = len(raw_feature)
            # 要将k*feature_len的原始特征转换为feature_len的特征
            pro_feature = [.0] * feature_len
            for i in range(feature_len):
                for j in range(k):
                    pro_feature[i] += raw_feature[j][i]
                pro_feature[i] /= k
            result.append(pro_feature)
        result = np.array(result)
        np.save(filename, result)
    return result