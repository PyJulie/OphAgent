# 第 9 章：扩展 OphAgent

新的眼科模型通过适配器层接入 OphAgent。正确实现一个适配器后，现有的会话、
规划器、执行器、缓存、核验器、Web UI 和导出流程就能使用这个新工具，无需在
中央循环中加入特定于该模型的逻辑。

本章说明所有必要的集成环节。

## 第 1 步：定义临床契约

编写代码前，先明确：

- 支持的模态；
- 临床任务；
- 所需输入与预处理；
- 输出标签或定量字段；
- 置信度定义与阈值；
- 已知失败模式；
- 所需上游工具；
- 预期运行成本；
- 生成的图像或掩膜。

这些信息属于 `ToolMetadata` 的组成部分，不是可有可无的文档说明。

## 第 2 步：创建适配器

下面是一个代码骨架。需要用真实实现替换占位的模型加载和推理逻辑。

```python
from pathlib import Path

from ophagent.adapters.base import (
    AdapterBase,
    AdapterResult,
    ToolMetadata,
    register,
)


@register
class ExampleCFPAdapter(AdapterBase):
    metadata = ToolMetadata(
        name="cfp_example_classifier",
        modality="CFP",
        task="classification",
        description=(
            "Classifies a CFP image for the example endpoint and returns "
            "calibrated class probabilities."
        ),
        input_size=(224, 224),
        labels=["negative", "positive"],
        confidence_threshold=0.60,
        limitations=[
            "Validated only on standard-field colour fundus photographs",
            "Do not use on UWF or FFA images",
        ],
        requires_tools=[],
        cost_class="fast",
    )

    def _load_impl(self) -> None:
        checkpoint = Path("/private/runtime/checkpoints/example/model.pt")
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")

        # Load the real model here.
        self._impl = load_example_model(checkpoint, device=self.device)

    def _predict_impl(self, image_path: str, **kwargs) -> AdapterResult:
        probabilities = run_example_inference(
            self._impl,
            image_path,
            device=self.device,
        )
        top_label = max(probabilities, key=probabilities.get)
        confidence = float(probabilities[top_label])

        return AdapterResult(
            success=True,
            tool=self.metadata.name,
            modality=self.metadata.modality,
            task=self.metadata.task,
            predictions={
                "top_label": top_label,
                "probabilities": probabilities,
            },
            confidence=confidence,
            metadata={
                "checkpoint_version": "example-v1",
            },
        )
```

不要在 `_predict_impl(...)` 内部捕获所有异常。`AdapterBase.predict` 已经会把意外
失败转换为结构化的失败 `AdapterResult`。

## 第 3 步：导入模块

Python 导入包含 `@register` 的模块时才会执行注册。需要把新模块加入相应的模态
包：

```python
# ophagent/adapters/cfp/__init__.py
from . import example_classifier  # noqa: F401
```

如果它确实属于可选依赖，可单独隔离该导入：

```python
try:
    from . import example_classifier  # noqa: F401
except Exception as exc:
    import logging
    logging.getLogger(__name__).warning(
        "Example CFP adapter not registered: %s", exc
    )
```

只有在确实允许部分功能不可用时才使用这种写法。必需适配器中的程序错误不应被
伪装成可选依赖缺失。

## 第 4 步：确认注册

```python
from ophagent.adapters import GLOBAL_REGISTRY

names = {tool.name for tool in GLOBAL_REGISTRY.list_tools("CFP")}
assert "cfp_example_classifier" in names
```

注册过程不应加载模型 checkpoint。第一次预测才应触发延迟加载：

```python
result = GLOBAL_REGISTRY.predict(
    "cfp_example_classifier",
    "tests/fixtures/example_cfp.jpg",
)

assert result.tool == "cfp_example_classifier"
assert isinstance(result.success, bool)
```

## 第 5 步：检查结果契约

至少需要测试：

1. 有效输入返回 `success=True`；
2. 置信度是含义明确且有文档说明的标量；
3. 低于阈值的结果会变为 `undetermined=True`；
4. 权重缺失时返回结构化失败；
5. 输入类型错误或文件不可读时安全失败；
6. 预测字段可以通过 `to_jsonable()` 序列化为 JSON；
7. 生成图像的路径保持在已配置的输出区域内；
8. 重复调用会复用已经加载的模型实例；
9. 工具元数据明确列出限制和必需依赖。

对于分割或检测工具，还要依据原始影像尺寸核对掩膜或边界框的几何关系。

## 第 6 步：整合选择行为

适配器注册后会自动出现在工具包中，但要获得合适的选择行为，可能仍需显式更新：

- 当多个工具服务于同一模态和任务时，在 `AdapterRegistry.tools_for(...)` 中增加
  首选使用顺序；
- 对组合工具，在 `cross_modal_tools_for(...)` 中增加跨模态要求；
- 只有当缺少某工具就无法安全完成该模态任务时，才把它设为核心工具；
- 通过 `requires_tools` 描述依赖关系；
- 更新 checkpoint 配置和预检覆盖。

不要把每个新工具都设为必需。冗余工具会增加延迟，也可能引入相互冲突的证据，
却不一定改善目标终点。

## 第 7 步：通过会话测试

直接测试适配器只能证明模型封装可用。完整的集成测试应经过 `OphSession`：

```python
from ophagent.chat.oph_session import OphSession

session = OphSession.new(
    backend="openai",
    model="gpt-5",
    effort="medium",
)
session.set_image("tests/fixtures/example_cfp.jpg")

reply = session.chat(
    "Use the available structured evidence to assess the example endpoint."
)

assert "cfp_example_classifier" in {
    tool_name
    for image_results in session.context.analyses.values()
    for tool_name in image_results
}
```

这项测试需要已配置的供应商，或受控的模型客户端测试夹具。测试还应确认核验器
能够读取新工具的结果。

## 扩展工作流

```mermaid
flowchart LR
    C["定义临床契约"] --> A["实现 AdapterBase 子类"]
    A --> I["导入并注册"]
    I --> R["注册表测试"]
    R --> P["预测与失败测试"]
    P --> S["OphSession 集成"]
    S --> V["核验器与导出检查"]
    V --> F["预检与文档"]
```

## 源码定位

| 职责 | 源码路径 |
|---|---|
| 适配器契约 | `ophagent/adapters/base.py` |
| 模态注册 | `ophagent/adapters/<modality>/__init__.py` |
| 首选工具排序 | `ophagent/adapters/base.py` |
| 工具包 schema 生成 | `ophagent/chat/oph_tools.py` |
| 核心证据与模态保护 | `ophagent/chat/oph_session.py` |
| 运行时资源配置 | `ophagent/checkpoint_config.py` |
| 部署验证 | `ophagent/preflight.py` |

## 小结

只有当模型封装、不确定性契约、注册、路由、证据缓存、核验器、测试和运行时资源
彼此一致时，OphAgent 扩展才算完整。适配器边界让这些集成要求保持明确，同时避免
中央循环与某一个模型实现耦合。

---

上一章：**[第 8 章——Web UI 与导出](08_web_ui_and_exports.md)**  
返回：**[教程目录](index.md)**
