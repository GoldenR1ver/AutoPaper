class AutoPaperError(Exception):
    """用户可以处理的错误。"""


class ConfigError(AutoPaperError):
    """配置文件或环境变量不正确。"""


class SourceError(AutoPaperError):
    """某个文献数据源请求失败。"""

    def __init__(self, source: str, message: str):
        self.source = source
        super().__init__(f"{source}: {message}")


class LLMError(AutoPaperError):
    """语言模型请求或解析失败。"""


class NotFoundError(AutoPaperError):
    """主题、论文或研究者不存在。"""


class AmbiguousError(AutoPaperError):
    """名称对应多条记录，需要用户指定。"""
