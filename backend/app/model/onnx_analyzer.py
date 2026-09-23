"""
ONNX Model Adapter and Graph Analyzer
"""
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import onnx
import onnxruntime as ort
from backend.app.model.loader import ModelAdapter

class ONNXModelAdapter(ModelAdapter):
    def __init__(self, model_path: Path):
        super().__init__(model_path)
        self.access_level = "GRAY_BOX"
        self.session: ort.InferenceSession = None
        self.onnx_model: onnx.ModelProto = None
        self.input_name: str = "input"
        self.input_shape: List[Any] = []
        self.output_name: str = "output"
        self.output_shape: List[Any] = []

    def load(self) -> bool:
        try:
            self.onnx_model = onnx.load(str(self.model_path))
            onnx.checker.check_model(self.onnx_model)
            
            # Setup ONNXRuntime session
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 2
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(str(self.model_path), opts, providers=["CPUExecutionProvider"])

            inputs = self.session.get_inputs()
            if inputs:
                self.input_name = inputs[0].name
                self.input_shape = inputs[0].shape

            outputs = self.session.get_outputs()
            if outputs:
                self.output_name = outputs[0].name
                self.output_shape = outputs[0].shape

            return True
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model {self.model_path.name}: {str(e)}")

    def predict(self, input_tensor: np.ndarray) -> np.ndarray:
        if self.session is None:
            self.load()

        # Ensure float32
        tensor = input_tensor.astype(np.float32)
        # Adapt dimensions if required (e.g. batch dimension)
        if len(tensor.shape) == 3:
            tensor = np.expand_dims(tensor, axis=0)

        # Transpose if expected NCHW vs NHWC
        if len(self.input_shape) == 4 and self.input_shape[1] in [1, 3] and tensor.shape[-1] in [1, 3]:
            # [B, H, W, C] -> [B, C, H, W]
            tensor = np.transpose(tensor, (0, 3, 1, 2))

        outputs = self.session.run([self.output_name], {self.input_name: tensor})
        return np.array(outputs[0])

    def inspect_architecture(self) -> Dict[str, Any]:
        if self.onnx_model is None:
            self.load()

        graph = self.onnx_model.graph
        nodes_count = len(graph.node)
        op_types = {}
        for n in graph.node:
            op_types[n.op_type] = op_types.get(n.op_type, 0) + 1

        param_count = 0
        for init in graph.initializer:
            shape = init.dims
            param_count += int(np.prod(shape))

        return {
            "framework": "ONNX",
            "opset_version": [op.version for op in self.onnx_model.opset_import],
            "producer_name": self.onnx_model.producer_name,
            "nodes_count": nodes_count,
            "op_types_summary": op_types,
            "parameter_count": param_count,
            "inputs": [{"name": i.name, "shape": i.shape, "type": i.type} for i in self.session.get_inputs()],
            "outputs": [{"name": o.name, "shape": o.shape, "type": o.type} for o in self.session.get_outputs()]
        }

    def compute_weight_fingerprint(self) -> Dict[str, Any]:
        if self.onnx_model is None:
            self.load()

        graph = self.onnx_model.graph
        if not graph.initializer:
            return {
                "available": False,
                "reason": "Weights are stored externally or graph has no initializers.",
                "layer_statistics": []
            }

        weight_stats = []
        overall_norms = []
        for init in graph.initializer:
            arr = onnx.numpy_helper.to_array(init)
            mean_val = float(np.mean(arr))
            std_val = float(np.std(arr))
            l2_norm = float(np.linalg.norm(arr))
            overall_norms.append(l2_norm)
            weight_stats.append({
                "name": init.name,
                "shape": list(arr.shape),
                "mean": round(mean_val, 5),
                "std": round(std_val, 5),
                "l2_norm": round(l2_norm, 5),
                "sparsity": round(float(np.sum(arr == 0) / arr.size), 4)
            })

        return {
            "available": True,
            "total_initializers": len(weight_stats),
            "mean_l2_norm": round(float(np.mean(overall_norms)), 5) if overall_norms else 0.0,
            "layer_statistics": weight_stats[:20]  # top 20 representative layers
        }
