import unittest
import numpy as np
from aggregate import fedavgm_aggregate

class TestFedAvgmAggregate(unittest.TestCase):

    def test_without_momentum(self):
        # 构造测试输入
        results = [
            ([np.array([1.1, 2.2]), np.array([3.3, 4.4])], 10),
            ([np.array([1.0, 2.0]), np.array([3.0, 4.0])], 20)
        ]
        previous_model = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        server_momentum = 0.0
        server_lr = 1.0

        # 调用待测试函数
        new_weights, pseudo_gradient = fedavgm_aggregate(results, previous_model, server_momentum, None, server_lr)

        # 预期输出
        expected_new_weights = [
            np.array([1.05, 2.1]),  # (1.1+1.0)/2 - 1.0
            np.array([3.15, 4.2])  # (3.3+3.0)/2 - 3.0
        ]
        expected_pseudo_gradient = [
            np.array([0.05, 0.1]),  # (1.1-1.0) + (1.0-1.0)
            np.array([0.15, 0.2])  # (3.3-3.0) + (3.0-3.0)
        ]

        # 测试权重是否正确更新
        for new_w, exp_new_w in zip(new_weights, expected_new_weights):
            np.testing.assert_array_almost_equal(new_w, exp_new_w)
        
        # 测试伪梯度是否正确计算
        for pg, exp_pg in zip(pseudo_gradient, expected_pseudo_gradient):
            np.testing.assert_array_almost_equal(pg, exp_pg)

    def test_with_momentum(self):
        # 构造测试输入
        results = [
            ([np.array([1.2, 2.4]), np.array([3.6, 4.8])], 10),
            ([np.array([1.1, 2.2]), np.array([3.3, 4.4])], 20)
        ]
        previous_model = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        server_momentum = 0.9
        momentum_vector = [
            np.array([0.1, 0.2]),
            np.array([0.3, 0.4])
        ]
        server_lr = 0.1

        # 调用待测试函数
        new_weights, pseudo_gradient = fedavgm_aggregate(results, previous_model, server_momentum, momentum_vector, server_lr)

        # 预期输出
        expected_new_weights = [
            np.array([0.97, 1.94]),  # (1.1*0.9 + 0.1) - (0.1*0.9 + 0.2) * 0.1
            np.array([2.91, 3.88])  # (3.3*0.9 + 0.3) - (0.3*0.9 + 0.4) * 0.1
        ]
        expected_pseudo_gradient = [
            np.array([0.9, 1.8]),  # (1.2-1.1)*0.9 + 0.1
            np.array([2.7, 3.6])  # (3.6-3.3)*0.9 + 0.3
        ]

        # 测试权重是否正确更新
        for new_w, exp_new_w in zip(new_weights, expected_new_weights):
            np.testing.assert_array_almost_equal(new_w, exp_new_w)
        
        # 测试伪梯度是否正确计算
        for pg, exp_pg in zip(pseudo_gradient, expected_pseudo_gradient):
            np.testing.assert_array_almost_equal(pg, exp_pg)

if __name__ == "__main__":
    unittest.main()
