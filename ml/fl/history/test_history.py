import unittest

# 假设以下是位于 history.py 中的 History 类的完整实现
class History:
    def __init__(self):
        self.global_test_losses = []

    def add_global_test_losses(self, averaged_loss: float) -> None:
        """Add one global test loss."""
        self.global_test_losses.append(averaged_loss)

# 单元测试类
class TestHistory(unittest.TestCase):
    def test_add_global_test_losses(self):
        # 创建 History 对象
        history = History()

        # 确认初始化时 global_test_losses 是一个空列表
        self.assertEqual(history.global_test_losses, [])

        # 添加一个测试损失
        averaged_loss = 0.123
        history.add_global_test_losses(averaged_loss)

        # 检查 global_test_losses 是否包含添加的损失值
        self.assertEqual(history.global_test_losses, [averaged_loss])

        # 添加另一个测试损失
        averaged_loss_2 = 0.456
        history.add_global_test_losses(averaged_loss_2)

        # 检查 global_test_losses 是否包含两个损失值
        self.assertEqual(history.global_test_losses, [averaged_loss, averaged_loss_2])

# 运行测试
if __name__ == "__main__":
    unittest.main()
