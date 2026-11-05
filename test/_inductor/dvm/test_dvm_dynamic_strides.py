import torch
from torch.fx import symbolic_trace
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from torch_npu._inductor.dvm.graph_build import is_fx_dynamic


def test_fixed_shape_with_symbolic_stride_requires_dynamic_kernel():
    symbolic_stride = ShapeEnv().create_unbacked_symint()

    class StridedTensor(torch.Tensor):
        def stride(self):
            return (symbolic_stride, 32, 1)

    value = torch.Tensor._make_subclass(StridedTensor, torch.empty(256, 1, 32), False)
    graph = symbolic_trace(lambda x: x.float())
    next(node for node in graph.graph.nodes if node.op == "placeholder").meta["val"] = value
    assert all(isinstance(size, int) for size in value.shape)
    assert is_fx_dynamic(graph)


def test_concrete_shape_and_strides_keep_static_kernel():
    graph = symbolic_trace(lambda x: x.float())
    next(node for node in graph.graph.nodes if node.op == "placeholder").meta["val"] = (
        torch.empty(256, 96)[:, 64:].unsqueeze(1)
    )
    assert not is_fx_dynamic(graph)
