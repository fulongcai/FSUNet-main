from typing import List
from mmengine.evaluator import BaseMetric
from mmengine.registry import METRICS
from .metrics import Metric


@METRICS.register_module()
class ISTDMetrics(BaseMetric):
    default_prefix = "ISTD"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metrics = []
        self.bins = 10
        # 初始化每个阈值的Metric对象
        for _ in range(self.bins):
            self.metrics.append(Metric())
        self.results = []  # 用来存储每个batch的结果

    def process(self, data_batch, data_samples):
        # 清空当前批次的计算结果
        batch_results = []

        # 遍历每个数据样本
        for i, result in enumerate(data_samples):
            y = result.unsqueeze(0)  # 假设每个 result 是模型输出
            gt = data_batch["gt"][i:i + 1, :].to(result.device)  # 获取 ground truth


            # 遍历每个阈值进行更新
            for i in range(self.bins):
                self.metrics[i].update(y > ((i + 1) / self.bins), gt > 0)

            # 每个样本的计算结果
            sample_result = {}
            for i in range(self.bins):
                metric_result = self.metrics[i].get()
                sample_result[f"Thres{((i + 1) / self.bins):1f}/iou%"] = metric_result[1] * 100
                sample_result[f"Thres{((i + 1) / self.bins):1f}/niou%"] = metric_result[2] * 100
                sample_result[f"Thres{((i + 1) / self.bins):1f}/Fa1e-6"] = metric_result[3] * 1e6
                sample_result[f"Thres{((i + 1) / self.bins):1f}/Pd%"] = metric_result[4] * 100

            # 将每个样本的结果保存到 batch_results
            batch_results.append(sample_result)

        # 将当前批次的结果添加到 self.results
        self.results.append(batch_results)

    def compute_metrics(self, results: List):
        """
        合并所有批次的结果，计算最终指标
        """
        ret = {}
        for i in range(self.bins):
            # 合并所有批次的结果
            total_metric_result = [0, 0, 0, 0, 0]  # 初始化一个结果累积数组
            count = 0

            for batch_result in results:
                for sample_result in batch_result:
                    metric_result = sample_result[f"Thres{((i + 1) / self.bins):1f}/iou%"]
                    total_metric_result[0] += metric_result
                    metric_result = sample_result[f"Thres{((i + 1) / self.bins):1f}/niou%"]
                    total_metric_result[1] += metric_result
                    metric_result = sample_result[f"Thres{((i + 1) / self.bins):1f}/Fa1e-6"]
                    total_metric_result[2] += metric_result
                    metric_result = sample_result[f"Thres{((i + 1) / self.bins):1f}/Pd%"]
                    total_metric_result[3] += metric_result
                    count += 1

            # 计算每个阈值的平均值
            for idx, metric_name in enumerate(
                    ['iou%', 'niou%', 'Fa1e-6', 'Pd%']
            ):
                ret[f"Thres{((i + 1) / self.bins):1f}/{metric_name}"] = total_metric_result[
                                                                            idx] / count if count > 0 else 0

        return ret
