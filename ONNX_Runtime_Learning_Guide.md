# Learning ONNX Runtime: A Comprehensive Guide

## Table of Contents

1. [Introduction to ONNX Runtime](#chapter-1-introduction-to-onnx-runtime)
2. [Understanding the Architecture](#chapter-2-understanding-the-architecture)
3. [Setting Up Your Development Environment](#chapter-3-setting-up-your-development-environment)
4. [Core Components Deep Dive](#chapter-4-core-components-deep-dive)
5. [Language Bindings and APIs](#chapter-5-language-bindings-and-apis)
6. [Execution Providers](#chapter-6-execution-providers)
7. [Building and Customizing ONNX Runtime](#chapter-7-building-and-customizing-onnx-runtime)
8. [Performance Optimization](#chapter-8-performance-optimization)
9. [Contributing to the Project](#chapter-9-contributing-to-the-project)
10. [Advanced Topics](#chapter-10-advanced-topics)

---

## Chapter 1: Introduction to ONNX Runtime

### What is ONNX Runtime?

ONNX Runtime is a cross-platform, high-performance machine learning inference and training accelerator developed by Microsoft. It serves as a runtime engine for models in the ONNX (Open Neural Network Exchange) format, which is an open standard for representing machine learning models.

### Key Features

1. **Cross-Platform**: Runs on Windows, Linux, macOS, and mobile platforms
2. **Multi-Language Support**: Python, C#, Java, JavaScript/TypeScript, C++, C, Objective-C
3. **Hardware Acceleration**: CPU, GPU (CUDA, DirectML), specialized accelerators
4. **Framework Agnostic**: Works with models from PyTorch, TensorFlow, scikit-learn, XGBoost, and more
5. **High Performance**: Optimized for both inference and training workloads

### ONNX Runtime's Role in the ML Ecosystem

```
[ML Framework] → [ONNX Model] → [ONNX Runtime] → [Hardware]
     ↓              ↓               ↓              ↓
  PyTorch        .onnx file    Optimizations   CPU/GPU/NPU
  TensorFlow                   Graph Transforms
  scikit-learn                 Memory Management
```

### Use Cases

- **Inference**: Deploy trained models for prediction in production
- **Training**: Accelerate model training on multi-node GPU clusters
- **Edge Deployment**: Run models on mobile devices and IoT
- **Cloud Services**: Scale ML workloads in cloud environments

---

## Chapter 2: Understanding the Architecture

### High-Level Architecture

ONNX Runtime follows a modular architecture designed for flexibility and performance:

```
┌─────────────────────────────────────────────────────────┐
│                 Language Bindings                      │
│  Python │ C# │ Java │ JavaScript │ C++ │ C │ Objective-C │
├─────────────────────────────────────────────────────────┤
│                   Core Runtime                         │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐ │
│  │   Session   │ │    Graph    │ │     Optimizer       │ │
│  │ Management  │ │   Engine    │ │     Engine          │ │
│  └─────────────┘ └─────────────┘ └─────────────────────┘ │
├─────────────────────────────────────────────────────────┤
│                Execution Providers                     │
│   CPU │ CUDA │ DirectML │ TensorRT │ OpenVINO │ ...     │
└─────────────────────────────────────────────────────────┘
```

### Core Components

#### 1. Session Management (`onnxruntime/core/session/`)
- **InferenceSession**: Main entry point for running inference
- **SessionOptions**: Configuration for runtime behavior
- **Environment**: Global runtime settings and initialization

#### 2. Graph Engine (`onnxruntime/core/graph/`)
- **Graph**: In-memory representation of the ONNX model
- **Node**: Individual operations in the graph
- **GraphViewer**: Read-only interface for graph inspection

#### 3. Framework (`onnxruntime/core/framework/`)
- **OpKernel**: Base class for operation implementations
- **ExecutionProvider**: Interface for hardware-specific implementations
- **MLValue/OrtValue**: Tensor and value containers

#### 4. Providers (`onnxruntime/core/providers/`)
- Hardware-specific implementations of operations
- CPU, CUDA, DirectML, TensorRT, OpenVINO, and many more

### Data Flow

1. **Model Loading**: ONNX model is parsed and converted to internal graph representation
2. **Graph Optimization**: Multiple optimization passes improve performance
3. **Provider Selection**: Operations are assigned to appropriate execution providers
4. **Execution**: Runtime executes the optimized graph with input data
5. **Output**: Results are returned through the chosen language binding

---

## Chapter 3: Setting Up Your Development Environment

### Prerequisites

- **Operating System**: Windows 10+, Ubuntu 18.04+, macOS 10.15+
- **Compilers**:
  - Windows: Visual Studio 2019+ or clang
  - Linux: GCC 7+ or clang 6+
  - macOS: Xcode 10+
- **Python**: 3.8+ (for Python development)
- **CMake**: 3.18+
- **Git**: For source code management

### Quick Start with Pre-built Packages

#### Python
```bash
pip install onnxruntime
# Or for GPU support
pip install onnxruntime-gpu
```

#### C#
```bash
dotnet add package Microsoft.ML.OnnxRuntime
# Or for GPU support
dotnet add package Microsoft.ML.OnnxRuntime.Gpu
```

#### JavaScript
```bash
npm install onnxruntime-web
# Or for Node.js
npm install onnxruntime-node
```

### Building from Source

#### 1. Clone the Repository
```bash
git clone https://github.com/microsoft/onnxruntime.git
cd onnxruntime
git submodule update --init --recursive
```

#### 2. Basic Build (CPU Only)
```bash
# Linux/macOS
./build.sh --config Release

# Windows
.\build.bat --config Release
```

#### 3. Build with GPU Support
```bash
# CUDA support
./build.sh --config Release --use_cuda --cuda_home /usr/local/cuda

# DirectML (Windows)
.\build.bat --config Release --use_dml
```

#### 4. Python Package Build
```bash
./build.sh --config Release --build_wheel
pip install build/Linux/Release/dist/*.whl
```

### Development Tools

#### Visual Studio Code Setup
1. Install C/C++ extension
2. Install Python extension
3. Install CMake Tools extension
4. Configure workspace settings for the project

#### Debugging Setup
- Use debug builds: `--config Debug`
- Set up breakpoints in your IDE
- Use runtime logging: `--enable_logging`

---

## Chapter 4: Core Components Deep Dive

### InferenceSession: The Heart of ONNX Runtime

The `InferenceSession` class is the primary interface for running inference:

```cpp
// C++ Example
#include "onnxruntime_cxx_api.h"

Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
Ort::SessionOptions session_options;
Ort::Session session(env, model_path, session_options);

// Run inference
auto output_tensors = session.Run(Ort::RunOptions{nullptr},
                                  input_names, input_tensors,
                                  output_names);
```

```python
# Python Example
import onnxruntime as ort

session = ort.InferenceSession("model.onnx")
outputs = session.run(None, {"input": input_data})
```

### Graph Representation

ONNX Runtime uses an internal graph representation that's optimized for execution:

#### Key Classes:
- **Graph**: Container for nodes, inputs, outputs, and initializers
- **Node**: Represents an operation (Add, Conv, etc.)
- **NodeArg**: Represents data flowing between nodes
- **GraphViewer**: Provides read-only access to graph structure

#### Graph Lifecycle:
1. **Loading**: ONNX protobuf → Graph object
2. **Validation**: Check graph consistency
3. **Optimization**: Apply transformation passes
4. **Partitioning**: Assign nodes to execution providers
5. **Execution**: Run the optimized graph

### Memory Management

ONNX Runtime uses sophisticated memory management:

#### Allocators:
- **CPUAllocator**: Standard CPU memory allocation
- **CUDAAllocator**: GPU memory management
- **ArenaAllocator**: Pool-based allocation for efficiency

#### Memory Patterns:
- **Input/Output Binding**: Reuse memory across runs
- **Memory Planning**: Pre-allocate memory for optimal performance
- **Memory Mapping**: Directly map model weights from disk

### Optimization Engine

The optimization engine applies multiple passes to improve performance:

#### Graph Optimizations:
1. **Constant Folding**: Compute constant expressions at load time
2. **Dead Code Elimination**: Remove unused nodes
3. **Operator Fusion**: Combine multiple operations
4. **Memory Layout Optimization**: Minimize data copying

#### Provider-Specific Optimizations:
- **CUDA**: Kernel fusion, memory coalescing
- **CPU**: SIMD vectorization, loop unrolling
- **TensorRT**: Dynamic shape optimization

---

## Chapter 5: Language Bindings and APIs

### Python API

The Python API is the most feature-complete and widely used:

#### Basic Usage:
```python
import onnxruntime as ort
import numpy as np

# Create session
session = ort.InferenceSession("model.onnx")

# Get input/output info
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Run inference
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
result = session.run([output_name], {input_name: input_data})
```

#### Advanced Features:
```python
# Custom execution providers
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
session = ort.InferenceSession("model.onnx", providers=providers)

# IO Binding for performance
io_binding = session.io_binding()
io_binding.bind_input('input', input_tensor)
io_binding.bind_output('output')
session.run_with_iobinding(io_binding)

# Session options
options = ort.SessionOptions()
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
options.enable_profiling = True
```

### C++ API

The C++ API provides the highest performance and most control:

#### Modern C++ Interface:
```cpp
#include "onnxruntime_cxx_api.h"

class ModelInference {
private:
    Ort::Env env_;
    Ort::Session session_;
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;

public:
    ModelInference(const std::string& model_path)
        : env_(ORT_LOGGING_LEVEL_WARNING, "ModelInference"),
          session_(env_, model_path.c_str(), Ort::SessionOptions{}) {

        // Get input/output names
        auto num_inputs = session_.GetInputCount();
        for (size_t i = 0; i < num_inputs; ++i) {
            input_names_.push_back(session_.GetInputName(i, allocator_));
        }
    }

    std::vector<Ort::Value> Run(const std::vector<Ort::Value>& inputs) {
        return session_.Run(Ort::RunOptions{nullptr},
                           input_names_.data(), inputs.data(), inputs.size(),
                           output_names_.data(), output_names_.size());
    }
};
```

### C# API

The C# API provides .NET integration:

```csharp
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

// Create session
var session = new InferenceSession("model.onnx");

// Prepare input
var inputTensor = new DenseTensor<float>(new[] { 1, 3, 224, 224 });
var inputs = new List<NamedOnnxValue>
{
    NamedOnnxValue.CreateFromTensor("input", inputTensor)
};

// Run inference
var outputs = session.Run(inputs);
var result = outputs.First().AsTensor<float>();
```

### JavaScript API

#### Web Browser:
```javascript
import * as ort from 'onnxruntime-web';

// Create session
const session = await ort.InferenceSession.create('model.onnx');

// Prepare input
const inputTensor = new ort.Tensor('float32', inputData, [1, 3, 224, 224]);

// Run inference
const outputs = await session.run({ input: inputTensor });
const result = outputs.output.data;
```

#### Node.js:
```javascript
const ort = require('onnxruntime-node');

async function runInference() {
    const session = await ort.InferenceSession.create('model.onnx');
    const input = new ort.Tensor('float32', inputData, [1, 3, 224, 224]);
    const outputs = await session.run({ input });
    return outputs.output.data;
}
```

### Java API

```java
import ai.onnxruntime.*;

// Create environment and session
OrtEnvironment env = OrtEnvironment.getEnvironment();
OrtSession session = env.createSession("model.onnx");

// Create input tensor
float[] inputData = new float[1 * 3 * 224 * 224];
OnnxTensor inputTensor = OnnxTensor.createTensor(env, inputData, new long[]{1, 3, 224, 224});

// Run inference
Map<String, OnnxTensor> inputs = Map.of("input", inputTensor);
OrtSession.Result outputs = session.run(inputs);
```

---

## Chapter 6: Execution Providers

Execution Providers (EPs) are the key to ONNX Runtime's performance and hardware support. They implement operations for specific hardware targets.

### Available Execution Providers

#### 1. CPU Execution Provider
- **Default provider**: Always available
- **Features**: SIMD optimizations, multi-threading
- **Use cases**: Development, CPU-only deployments

```python
# Explicitly use CPU
session = ort.InferenceSession("model.onnx",
                              providers=['CPUExecutionProvider'])
```

#### 2. CUDA Execution Provider
- **Hardware**: NVIDIA GPUs
- **Requirements**: CUDA 11.4+ and cuDNN 8.2+
- **Features**: Kernel fusion, memory optimization

```python
# CUDA with options
cuda_options = {
    'device_id': 0,
    'arena_extend_strategy': 'kNextPowerOfTwo',
    'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB
    'cudnn_conv_algo_search': 'EXHAUSTIVE',
}
session = ort.InferenceSession("model.onnx",
                              providers=[('CUDAExecutionProvider', cuda_options)])
```

#### 3. TensorRT Execution Provider
- **Hardware**: NVIDIA GPUs with TensorRT
- **Features**: Advanced optimizations, INT8 quantization
- **Best for**: Production inference with NVIDIA hardware

```python
trt_options = {
    'device_id': 0,
    'trt_max_workspace_size': 1 << 30,  # 1GB
    'trt_fp16_enable': True,
    'trt_int8_enable': False,
}
session = ort.InferenceSession("model.onnx",
                              providers=[('TensorrtExecutionProvider', trt_options)])
```

#### 4. DirectML Execution Provider
- **Platform**: Windows with DirectX 12
- **Hardware**: Any DirectX 12 compatible GPU
- **Features**: Cross-vendor GPU support

```python
# DirectML
session = ort.InferenceSession("model.onnx",
                              providers=['DmlExecutionProvider'])
```

#### 5. OpenVINO Execution Provider
- **Hardware**: Intel CPUs, GPUs, VPUs
- **Features**: Optimized for Intel hardware
- **Use cases**: Edge devices, Intel-based systems

```python
openvino_options = {
    'device_type': 'CPU_FP32',
    'precision': 'FP16',
    'num_of_threads': 4,
}
session = ort.InferenceSession("model.onnx",
                              providers=[('OpenVINOExecutionProvider', openvino_options)])
```

### Provider Fallback Mechanism

ONNX Runtime uses a fallback mechanism:

```python
# Provider priority order
providers = [
    'TensorrtExecutionProvider',  # Highest priority
    'CUDAExecutionProvider',      # Fallback 1
    'CPUExecutionProvider'        # Fallback 2 (always available)
]
session = ort.InferenceSession("model.onnx", providers=providers)
```

### Custom Execution Providers

You can create custom execution providers for specialized hardware:

```cpp
// Custom EP interface
class CustomExecutionProvider : public IExecutionProvider {
public:
    std::vector<std::unique_ptr<ComputeCapability>>
    GetCapability(const GraphViewer& graph_viewer,
                  const IKernelLookup& kernel_lookup) const override;

    Status Compile(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
                   std::vector<NodeComputeInfo>& node_compute_funcs) override;
};
```

### Performance Considerations

#### Provider Selection Guidelines:
1. **TensorRT**: Best for NVIDIA GPU production inference
2. **CUDA**: Good for NVIDIA GPU development and flexibility
3. **DirectML**: Best for Windows cross-vendor GPU support
4. **OpenVINO**: Optimal for Intel hardware
5. **CPU**: Universal fallback, good for development

#### Benchmarking Providers:
```python
import time

def benchmark_provider(model_path, provider, num_runs=100):
    session = ort.InferenceSession(model_path, providers=[provider])
    input_name = session.get_inputs()[0].name

    # Warmup
    for _ in range(10):
        session.run(None, {input_name: input_data})

    # Benchmark
    start_time = time.time()
    for _ in range(num_runs):
        session.run(None, {input_name: input_data})

    avg_time = (time.time() - start_time) / num_runs
    return avg_time
```

---

## Chapter 7: Building and Customizing ONNX Runtime

### Build System Overview

ONNX Runtime uses CMake as its primary build system, with Python scripts to orchestrate the build process.

### Build Script (`build.py`)

The main build script is located at `tools/ci_build/build.py`:

```bash
# Basic CPU build
python tools/ci_build/build.py --config Release

# GPU build with CUDA
python tools/ci_build/build.py --config Release --use_cuda --cuda_home /usr/local/cuda

# Build with multiple providers
python tools/ci_build/build.py --config Release --use_cuda --use_dnnl --use_openvino
```

### Common Build Options

#### Performance Options:
```bash
# Enable all optimizations
--enable_lto                    # Link-time optimization
--use_openmp                    # OpenMP parallelization
--use_mlas                      # Microsoft Linear Algebra Subprograms

# Memory optimizations
--enable_memory_profiling       # Memory usage profiling
--use_jemalloc                  # Alternative memory allocator
```

#### Provider Options:
```bash
# Execution providers
--use_cuda                      # NVIDIA CUDA
--use_tensorrt                  # NVIDIA TensorRT
--use_dml                       # DirectML (Windows)
--use_openvino                  # Intel OpenVINO
--use_dnnl                      # Intel DNNL (oneDNN)
--use_nnapi                     # Android NNAPI
--use_coreml                    # Apple CoreML
```

#### Platform-Specific Options:
```bash
# Mobile platforms
--android                       # Android build
--ios                          # iOS build
--osx                          # macOS build

# WebAssembly
--build_wasm                   # WebAssembly build
--enable_wasm_simd             # SIMD support for WASM
```

### Custom Builds

#### Minimal Build for Specific Models:
```bash
# Build only required operators
python tools/ci_build/build.py \
    --config Release \
    --include_ops_by_config your_model_ops.config \
    --minimal_build extended
```

#### Custom Operator Build:
```bash
# Include custom operators
python tools/ci_build/build.py \
    --config Release \
    --use_custom_op_library \
    --custom_op_library_path /path/to/custom_ops
```

### Cross-Compilation

#### ARM64 Linux:
```bash
python tools/ci_build/build.py \
    --config Release \
    --arm64 \
    --cmake_extra_defines CMAKE_TOOLCHAIN_FILE=cmake/linux_arm64_crosscompile_toolchain.cmake
```

#### Android:
```bash
python tools/ci_build/build.py \
    --config Release \
    --android \
    --android_api 29 \
    --android_ndk_path $ANDROID_NDK_ROOT
```

### Python Package Customization

#### Custom Python Build:
```bash
# Build wheel with specific features
python setup.py bdist_wheel \
    --build-dir build/Linux/Release \
    --use_cuda \
    --cuda_home /usr/local/cuda
```

#### Development Installation:
```bash
# Editable installation for development
pip install -e . --config-settings build-dir=build/Linux/Release
```

### Docker Builds

ONNX Runtime provides Docker containers for consistent builds:

```dockerfile
# Use official build container
FROM mcr.microsoft.com/onnxruntime/onnxruntime:latest-cuda

# Or build from source in container
FROM nvidia/cuda:11.8-devel-ubuntu20.04
RUN git clone https://github.com/microsoft/onnxruntime.git
WORKDIR onnxruntime
RUN ./build.sh --config Release --use_cuda
```

### Build Optimization Tips

#### Parallel Builds:
```bash
# Use all available cores
python tools/ci_build/build.py --config Release --parallel $(nproc)
```

#### Incremental Builds:
```bash
# Skip submodule updates for faster rebuilds
python tools/ci_build/build.py --config Release --skip_submodule_sync
```

#### Debug Builds:
```bash
# Build with debug symbols
python tools/ci_build/build.py --config Debug --enable_logging
```

---

## Chapter 8: Performance Optimization

### Understanding Performance Bottlenecks

#### Common Performance Issues:
1. **Memory allocation overhead**
2. **Data layout inefficiencies**
3. **Suboptimal operator fusion**
4. **Incorrect provider selection**
5. **Graph optimization limitations**

### Profiling and Monitoring

#### Built-in Profiling:
```python
# Enable profiling
options = ort.SessionOptions()
options.enable_profiling = True
session = ort.InferenceSession("model.onnx", options)

# Run inference
session.run(None, inputs)

# Get profiling results
prof_file = session.end_profiling()
print(f"Profiling data saved to: {prof_file}")
```

#### Memory Profiling:
```python
# Monitor memory usage
options = ort.SessionOptions()
options.enable_mem_pattern = True
options.enable_mem_reuse = True
session = ort.InferenceSession("model.onnx", options)
```

### Graph Optimization Theory and Implementation

Graph optimization is a critical component that transforms the computational graph to improve performance while preserving mathematical equivalence. ONNX Runtime employs multiple optimization strategies at different levels.

#### Optimization Levels and Strategies

```python
# Set optimization level
options = ort.SessionOptions()
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
# Options: ORT_DISABLE_ALL, ORT_ENABLE_BASIC, ORT_ENABLE_EXTENDED, ORT_ENABLE_ALL
```

#### 1. Basic Optimizations (Level 1)

**Constant Folding**: Pre-compute operations with constant inputs
```cpp
// Example: Conv → BatchNorm fusion with constant folding
class ConstantFoldingTransformer : public GraphTransformer {
 public:
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level) const override {
    for (auto& node : graph.Nodes()) {
      if (node.OpType() == "Add" && AllInputsAreConstants(node)) {
        // Fold constant addition at graph construction time
        auto result = ComputeConstantAdd(node);
        ReplaceNodeWithConstant(graph, node, result);
        modified = true;
      }
    }
    return Status::OK();
  }
};
```

**Dead Code Elimination**: Remove unreachable or unused nodes
```cpp
class DeadCodeEliminationTransformer : public GraphTransformer {
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level) const override {
    std::unordered_set<NodeIndex> reachable_nodes;

    // Mark all nodes reachable from graph outputs
    for (const auto* output : graph.GetOutputs()) {
      MarkReachableNodes(graph, output, reachable_nodes);
    }

    // Remove unreachable nodes
    auto nodes_to_remove = GetUnreachableNodes(graph, reachable_nodes);
    for (auto node_index : nodes_to_remove) {
      graph.RemoveNode(node_index);
      modified = true;
    }

    return Status::OK();
  }
};
```

#### 2. Extended Optimizations (Level 2)

**Operator Fusion**: Combine multiple operations into single kernels

#### Comprehensive Fusion Patterns

#### A. Convolution-Based Fusions

```cpp
// 1. Conv + BatchNorm + ReLU fusion
class ConvBNReluFusion : public GraphTransformer {
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level) const override {
    auto conv_nodes = graph.GetNodesOfOpType("Conv");

    for (auto* conv_node : conv_nodes) {
      if (auto fusion_opportunity = AnalyzeConvFusion(*conv_node)) {
        FuseConvBatchNormActivation(graph, fusion_opportunity);
        modified = true;
      }
    }
    return Status::OK();
  }
};

// 2. Conv + Add (Residual connection) fusion
class ConvAddFusion : public GraphTransformer {
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level) const override {
    for (auto& conv_node : graph.Nodes()) {
      if (conv_node.OpType() == "Conv") {
        // Look for Add node that consumes Conv output
        auto add_node = FindResidualAdd(conv_node);
        if (add_node && CanFuseConvAdd(conv_node, *add_node)) {
          CreateFusedConvAdd(graph, conv_node, *add_node);
          modified = true;
        }
      }
    }
    return Status::OK();
  }
};

// 3. Depthwise Conv + Pointwise Conv (MobileNet pattern) fusion
class DepthwisePointwiseFusion : public GraphTransformer {
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level) const override {
    for (auto& dw_conv : graph.Nodes()) {
      if (IsDepthwiseConv(dw_conv)) {
        auto pw_conv = FindFollowingPointwiseConv(dw_conv);
        if (pw_conv && CanFuseDepthwisePointwise(dw_conv, *pw_conv)) {
          CreateSeparableConvKernel(graph, dw_conv, *pw_conv);
          modified = true;
        }
      }
    }
    return Status::OK();
  }
};

// 4. Conv + Transpose (for upsampling) fusion
class ConvTransposeFusion : public GraphTransformer {
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level) const override {
    for (auto& conv_node : graph.Nodes()) {
      if (conv_node.OpType() == "Conv") {
        auto transpose_node = FindFollowingTranspose(conv_node);
        if (transpose_node && IsUpsamplingPattern(conv_node, *transpose_node)) {
          FuseConvTranspose(graph, conv_node, *transpose_node);
          modified = true;
        }
      }
    }
    return Status::OK();
  }
};
```

#### B. Activation Function Fusions

```cpp
// 1. Comprehensive activation fusion
class ActivationFusionTransformer : public GraphTransformer {
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level) const override {
    // GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    FuseGELUApproximation(graph, modified);

    // Swish/SiLU: x * sigmoid(x)
    FuseSwishActivation(graph, modified);

    // Mish: x * tanh(softplus(x))
    FuseMishActivation(graph, modified);

    // LeakyReLU: max(x, alpha * x)
    FuseLeakyReLU(graph, modified);

    // PReLU: max(x, alpha * x) where alpha is learnable
    FusePReLU(graph, modified);

    return Status::OK();
  }

 private:
  void FuseGELUApproximation(Graph& graph, bool& modified) {
    // Pattern: x * 0.5 * (1 + Tanh(Sqrt(2/Pi) * (x + 0.044715 * Pow(x, 3))))
    for (auto& mul_node : graph.Nodes()) {
      if (mul_node.OpType() == "Mul") {
        auto gelu_pattern = DetectGELUPattern(mul_node);
        if (gelu_pattern.is_valid) {
          CreateFusedGELU(graph, gelu_pattern);
          modified = true;
        }
      }
    }
  }

  void FuseSwishActivation(Graph& graph, bool& modified) {
    // Pattern: x * Sigmoid(x)
    for (auto& mul_node : graph.Nodes()) {
      if (mul_node.OpType() == "Mul") {
        auto swish_pattern = DetectSwishPattern(mul_node);
        if (swish_pattern.is_valid) {
          CreateFusedSwish(graph, swish_pattern);
          modified = true;
        }
      }
    }
  }
};
```

#### C. Normalization Fusions

```cpp
// 1. LayerNorm fusion (critical for transformers)
class LayerNormFusion : public GraphTransformer {
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level) const override {
    // Pattern: (x - mean) / sqrt(variance + epsilon) * scale + bias
    for (auto& sub_node : graph.Nodes()) {
      if (sub_node.OpType() == "Sub") {
        auto ln_pattern = DetectLayerNormPattern(sub_node);
        if (ln_pattern.is_complete) {
          CreateFusedLayerNorm(graph, ln_pattern);
          modified = true;
        }
      }
    }
    return Status::OK();
  }

 private:
  struct LayerNormPattern {
    Node* sub_node;          // x - mean
    Node* reduce_mean_node;  // mean calculation
    Node* div_node;          // division by std
    Node* sqrt_node;         // sqrt(variance)
    Node* add_epsilon_node;  // variance + epsilon
    Node* reduce_var_node;   // variance calculation
    Node* mul_scale_node;    // multiply by scale
    Node* add_bias_node;     // add bias
    bool is_complete = false;
  };
};

// 2. GroupNorm fusion
class GroupNormFusion : public GraphTransformer {
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level) const override {
    // Pattern similar to LayerNorm but with group dimension
    for (auto& reshape_node : graph.Nodes()) {
      if (reshape_node.OpType() == "Reshape") {
        auto gn_pattern = DetectGroupNormPattern(reshape_node);
        if (gn_pattern.is_valid) {
          CreateFusedGroupNorm(graph, gn_pattern);
          modified = true;
        }
      }
    }
    return Status::OK();
  }
};

// 3. InstanceNorm fusion
class InstanceNormFusion : public GraphTransformer {
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level) const override {
    // Pattern: per-instance normalization
    for (auto& reduce_node : graph.Nodes()) {
      if (reduce_node.OpType() == "ReduceMean") {
        auto in_pattern = DetectInstanceNormPattern(reduce_node);
        if (in_pattern.is_valid) {
          CreateFusedInstanceNorm(graph, in_pattern);
          modified = true;
        }
      }
    }
    return Status::OK();
  }
};
```

#### D. Attention Mechanism Fusions

```cpp
// 1. Multi-Head Attention fusion (complete)
class MultiHeadAttentionFusion : public GraphTransformer {
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level) const override {
    for (auto& matmul_node : graph.Nodes()) {
      if (matmul_node.OpType() == "MatMul") {
        auto attention_pattern = DetectFullAttentionPattern(matmul_node);
        if (attention_pattern.is_complete) {
          CreateOptimizedAttention(graph, attention_pattern);
          modified = true;
        }
      }
    }
    return Status::OK();
  }

 private:
  struct AttentionPattern {
    // Input projections
    Node* q_linear;
    Node* k_linear;
    Node* v_linear;

    // Attention computation
    Node* qk_matmul;        // Q @ K^T
    Node* scale_div;        // / sqrt(d_k)
    Node* mask_add;         // + attention_mask (optional)
    Node* softmax;          // softmax(scores)
    Node* dropout;          // dropout (optional)
    Node* av_matmul;        // @ V

    // Output projection
    Node* output_linear;
    Node* output_dropout;   // (optional)

    // Multi-head specific
    Node* q_reshape;        // reshape for multi-head
    Node* k_reshape;
    Node* v_reshape;
    Node* output_reshape;   // reshape back

    bool is_complete = false;
    int num_heads;
    int head_dim;
  };
};

// 2. Self-Attention with relative position encoding
class RelativePositionAttentionFusion : public GraphTransformer {
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level) const override {
    for (auto& embedding_node : graph.Nodes()) {
      if (embedding_node.OpType() == "Gather") {
        auto rel_pos_pattern = DetectRelativePositionPattern(embedding_node);
        if (rel_pos_pattern.is_valid) {
          CreateFusedRelativeAttention(graph, rel_pos_pattern);
          modified = true;
        }
      }
    }
    return Status::OK();
  }
};

// 3. Cross-Attention fusion (encoder-decoder)
class CrossAttentionFusion : public GraphTransformer {
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level) const override {
    // Q from decoder, K,V from encoder
    for (auto& cross_attn : graph.Nodes()) {
      if (IsCrossAttentionPattern(cross_attn)) {
        auto pattern = AnalyzeCrossAttentionPattern(cross_attn);
        CreateFusedCrossAttention(graph, pattern);
        modified = true;
      }
    }
    return Status::OK();
  }
};
```

#### E. Feed-Forward Network Fusions

```cpp
// 1. Transformer FFN fusion
class TransformerFFNFusion : public GraphTransformer {
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level) const override {
    // Pattern: Linear → Activation → Linear
    for (auto& linear1 : graph.Nodes()) {
      if (linear1.OpType() == "MatMul") {
        auto ffn_pattern = DetectFFNPattern(linear1);
        if (ffn_pattern.is_complete) {
          CreateFusedFFN(graph, ffn_pattern);
          modified = true;
        }
      }
    }
    return Status::OK();
  }

 private:
  struct FFNPattern {
    Node* linear1;          // First linear layer
    Node* activation;       // GELU, ReLU, Swish, etc.
    Node* dropout1;         // Optional dropout
    Node* linear2;          // Second linear layer
    Node* dropout2;         // Optional dropout
    bool is_complete = false;
  };
};

// 2. GLU (Gated Linear Unit) fusion
class GLUFusion : public GraphTransformer {
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level) const override {
    // Pattern: Linear → Split → [Gate, Value] → Sigmoid(Gate) * Value
    for (auto& split_node : graph.Nodes()) {
      if (split_node.OpType() == "Split") {
        auto glu_pattern = DetectGLUPattern(split_node);
        if (glu_pattern.is_valid) {
          CreateFusedGLU(graph, glu_pattern);
          modified = true;
        }
      }
    }
    return Status::OK();
  }
};

// 3. SwiGLU fusion (used in modern language models)
class SwiGLUFusion : public GraphTransformer {
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level) const override {
    // Pattern: Swish(Linear(x)) * Linear(x)
    for (auto& mul_node : graph.Nodes()) {
      if (mul_node.OpType() == "Mul") {
        auto swiglu_pattern = DetectSwiGLUPattern(mul_node);
        if (swiglu_pattern.is_valid) {
          CreateFusedSwiGLU(graph, swiglu_pattern);
          modified = true;
        }
      }
    }
    return Status::OK();
  }
};
```

#### F. Embedding and Positional Encoding Fusions

```cpp
// 1. Token + Position embedding fusion
class EmbeddingFusion : public GraphTransformer {
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level) const override {
    for (auto& token_embed : graph.Nodes()) {
      if (token_embed.OpType() == "Gather") {
        auto embed_pattern = DetectEmbeddingPattern(token_embed);
        if (embed_pattern.has_positional) {
          CreateFusedEmbedding(graph, embed_pattern);
          modified = true;
        }
      }
    }
    return Status::OK();
  }

 private:
  struct EmbeddingPattern {
    Node* token_embedding;
    Node* position_embedding;
    Node* add_embeddings;
    Node* layer_norm;         // Optional
    Node* dropout;            // Optional
    bool has_positional = false;
  };
};

// 2. Rotary Position Embedding (RoPE) fusion
class RoPEFusion : public GraphTransformer {
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level) const override {
    // Pattern: Complex rotation using sin/cos
    for (auto& reshape_node : graph.Nodes()) {
      if (reshape_node.OpType() == "Reshape") {
        auto rope_pattern = DetectRoPEPattern(reshape_node);
        if (rope_pattern.is_valid) {
          CreateFusedRoPE(graph, rope_pattern);
          modified = true;
        }
      }
    }
    return Status::OK();
  }
};
```

#### G. Loss Function Fusions

```cpp
// 1. Softmax + CrossEntropy fusion
class SoftmaxCrossEntropyFusion : public GraphTransformer {
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level) const override {
    for (auto& softmax_node : graph.Nodes()) {
      if (softmax_node.OpType() == "Softmax") {
        auto ce_node = FindCrossEntropyConsumer(softmax_node);
        if (ce_node) {
          CreateFusedSoftmaxCrossEntropy(graph, softmax_node, *ce_node);
          modified = true;
        }
      }
    }
    return Status::OK();
  }
};

// 2. Log + Softmax fusion (LogSoftmax)
class LogSoftmaxFusion : public GraphTransformer {
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level) const override {
    for (auto& log_node : graph.Nodes()) {
      if (log_node.OpType() == "Log") {
        auto softmax_input = FindSoftmaxInput(log_node);
        if (softmax_input && softmax_input->OpType() == "Softmax") {
          CreateFusedLogSoftmax(graph, *softmax_input, log_node);
          modified = true;
        }
      }
    }
    return Status::OK();
  }
};
```

#### H. Recurrent Network Fusions

```cpp
// 1. LSTM cell fusion
class LSTMCellFusion : public GraphTransformer {
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level) const override {
    // Pattern: Input gate, Forget gate, Output gate, Cell state computations
    for (auto& sigmoid_node : graph.Nodes()) {
      if (sigmoid_node.OpType() == "Sigmoid") {
        auto lstm_pattern = DetectLSTMCellPattern(sigmoid_node);
        if (lstm_pattern.is_complete) {
          CreateFusedLSTMCell(graph, lstm_pattern);
          modified = true;
        }
      }
    }
    return Status::OK();
  }

 private:
  struct LSTMCellPattern {
    Node* input_gate;
    Node* forget_gate;
    Node* output_gate;
    Node* cell_candidate;
    Node* cell_state_update;
    Node* hidden_state_update;
    bool is_complete = false;
  };
};

// 2. GRU cell fusion
class GRUCellFusion : public GraphTransformer {
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level) const override {
    // Pattern: Reset gate, Update gate, New gate
    for (auto& sigmoid_node : graph.Nodes()) {
      if (sigmoid_node.OpType() == "Sigmoid") {
        auto gru_pattern = DetectGRUCellPattern(sigmoid_node);
        if (gru_pattern.is_complete) {
          CreateFusedGRUCell(graph, gru_pattern);
          modified = true;
        }
      }
    }
    return Status::OK();
  }
};
```

#### I. Computer Vision Specific Fusions

```cpp
// 1. Non-Maximum Suppression (NMS) fusion
class NMSFusion : public GraphTransformer {
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level) const override {
    // Pattern used in object detection models
    for (auto& gather_node : graph.Nodes()) {
      if (gather_node.OpType() == "Gather") {
        auto nms_pattern = DetectNMSPattern(gather_node);
        if (nms_pattern.is_valid) {
          CreateFusedNMS(graph, nms_pattern);
          modified = true;
        }
      }
    }
    return Status::OK();
  }
};

// 2. ROI Pooling fusion (for R-CNN variants)
class ROIPoolingFusion : public GraphTransformer {
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level) const override {
    for (auto& slice_node : graph.Nodes()) {
      if (slice_node.OpType() == "Slice") {
        auto roi_pattern = DetectROIPoolingPattern(slice_node);
        if (roi_pattern.is_valid) {
          CreateFusedROIPooling(graph, roi_pattern);
          modified = true;
        }
      }
    }
    return Status::OK();
  }
};

// 3. Focal Loss fusion
class FocalLossFusion : public GraphTransformer {
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level) const override {
    // Pattern: (1 - p_t)^γ * CE(p_t, y_t)
    for (auto& pow_node : graph.Nodes()) {
      if (pow_node.OpType() == "Pow") {
        auto focal_pattern = DetectFocalLossPattern(pow_node);
        if (focal_pattern.is_valid) {
          CreateFusedFocalLoss(graph, focal_pattern);
          modified = true;
        }
      }
    }
    return Status::OK();
  }
};
```

#### J. Matrix Operation Fusions

```cpp
// 1. Matrix chain multiplication optimization
class MatMulChainFusion : public GraphTransformer {
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level) const override {
    // Optimize A @ B @ C sequences
    for (auto& matmul_node : graph.Nodes()) {
      if (matmul_node.OpType() == "MatMul") {
        auto chain = FindMatMulChain(matmul_node);
        if (chain.size() > 2) {
          OptimizeMatMulChain(graph, chain);
          modified = true;
        }
      }
    }
    return Status::OK();
  }
};

// 2. Batch MatMul with broadcast fusion
class BatchMatMulFusion : public GraphTransformer {
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level) const override {
    for (auto& expand_node : graph.Nodes()) {
      if (expand_node.OpType() == "Expand") {
        auto bmm_pattern = DetectBatchMatMulPattern(expand_node);
        if (bmm_pattern.is_valid) {
          CreateFusedBatchMatMul(graph, bmm_pattern);
          modified = true;
        }
      }
    }
    return Status::OK();
  }
};

// 3. Einsum optimization
class EinsumFusion : public GraphTransformer {
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level) const override {
    for (auto& einsum_node : graph.Nodes()) {
      if (einsum_node.OpType() == "Einsum") {
        auto optimized_einsum = OptimizeEinsum(einsum_node);
        if (optimized_einsum.is_optimized) {
          ReplaceEinsumWithOptimized(graph, einsum_node, optimized_einsum);
          modified = true;
        }
      }
    }
    return Status::OK();
  }
};
```

#### K. Graph-Level Optimizations

```cpp
// 1. Subgraph substitution
class SubgraphSubstitution : public GraphTransformer {
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level) const override {
    // Replace common subgraphs with optimized implementations
    auto subgraph_matches = FindCommonSubgraphs(graph);

    for (const auto& match : subgraph_matches) {
      if (HasOptimizedImplementation(match.pattern)) {
        ReplaceSubgraph(graph, match, GetOptimizedImplementation(match.pattern));
        modified = true;
      }
    }

    return Status::OK();
  }
};

// 2. Loop optimization
class LoopOptimization : public GraphTransformer {
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level) const override {
    // Optimize loops by unrolling, vectorization, etc.
    auto loops = DetectLoops(graph);

    for (const auto& loop : loops) {
      if (ShouldUnrollLoop(loop)) {
        UnrollLoop(graph, loop);
        modified = true;
      } else if (CanVectorizeLoop(loop)) {
        VectorizeLoop(graph, loop);
        modified = true;
      }
    }

    return Status::OK();
  }
};
```

**Memory Layout Optimization**: Minimize data copying and improve cache locality
```cpp
class MemoryLayoutOptimizer : public GraphTransformer {
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level) const override {
    // Analyze memory access patterns
    auto memory_analysis = AnalyzeMemoryPatterns(graph);

    // Insert transpose operations to optimize data layout
    for (const auto& layout_change : memory_analysis.recommended_changes) {
      if (layout_change.benefit_score > threshold) {
        InsertLayoutTransformation(graph, layout_change);
        modified = true;
      }
    }

    return Status::OK();
  }
};
```

#### 3. All Optimizations (Level 3)

**Advanced Pattern Matching**: Complex multi-node patterns
```cpp
// Attention pattern optimization
class AttentionOptimizer : public GraphTransformer {
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level) const override {
    // Pattern: MatMul → Add → Softmax → MatMul (Multi-head attention)
    auto attention_patterns = FindAttentionPatterns(graph);

    for (const auto& pattern : attention_patterns) {
      if (CanOptimizeAttention(pattern)) {
        // Replace with optimized attention kernel
        auto optimized_attention = CreateOptimizedAttention(pattern);
        ReplacePattern(graph, pattern, optimized_attention);
        modified = true;
      }
    }

    return Status::OK();
  }

 private:
  struct AttentionPattern {
    Node* query_matmul;
    Node* key_matmul;
    Node* value_matmul;
    Node* scale_node;
    Node* softmax_node;
    Node* output_matmul;
  };
};
```

#### Graph Optimization Pipeline

```python
# Custom optimization pipeline with comprehensive fusion support
class ComprehensiveOptimizationPipeline:
    def __init__(self):
        # Organize transformers by execution order and dependencies
        self.basic_transformers = [
            ConstantFoldingTransformer(),
            DeadCodeEliminationTransformer(),
            CommonSubexpressionEliminationTransformer(),
        ]

        # Convolution-based fusions
        self.conv_transformers = [
            ConvBNReluFusion(),
            ConvAddFusion(),
            DepthwisePointwiseFusion(),
            ConvTransposeFusion(),
        ]

        # Activation and normalization fusions
        self.activation_norm_transformers = [
            ActivationFusionTransformer(),
            LayerNormFusion(),
            GroupNormFusion(),
            InstanceNormFusion(),
        ]

        # Attention and transformer optimizations
        self.attention_transformers = [
            MultiHeadAttentionFusion(),
            RelativePositionAttentionFusion(),
            CrossAttentionFusion(),
            TransformerFFNFusion(),
            GLUFusion(),
            SwiGLUFusion(),
        ]

        # Embedding and positional encoding
        self.embedding_transformers = [
            EmbeddingFusion(),
            RoPEFusion(),
        ]

        # Loss function optimizations
        self.loss_transformers = [
            SoftmaxCrossEntropyFusion(),
            LogSoftmaxFusion(),
            FocalLossFusion(),
        ]

        # Recurrent network optimizations
        self.rnn_transformers = [
            LSTMCellFusion(),
            GRUCellFusion(),
        ]

        # Computer vision optimizations
        self.cv_transformers = [
            NMSFusion(),
            ROIPoolingFusion(),
        ]

        # Matrix operation optimizations
        self.matrix_transformers = [
            MatMulChainFusion(),
            BatchMatMulFusion(),
            EinsumFusion(),
        ]

        # High-level graph optimizations
        self.graph_transformers = [
            SubgraphSubstitution(),
            LoopOptimization(),
            MemoryLayoutOptimizer(),
        ]

    def optimize_graph(self, graph, target_model_type="transformer"):
        """
        Optimize graph based on model type

        Args:
            graph: ONNX graph to optimize
            target_model_type: "transformer", "cnn", "rnn", or "mixed"
        """
        modified = True
        iteration = 0
        max_iterations = 15

        # Define optimization stages based on model type
        if target_model_type == "transformer":
            stages = [
                self.basic_transformers,
                self.activation_norm_transformers,
                self.attention_transformers,
                self.embedding_transformers,
                self.loss_transformers,
                self.matrix_transformers,
                self.graph_transformers
            ]
        elif target_model_type == "cnn":
            stages = [
                self.basic_transformers,
                self.conv_transformers,
                self.activation_norm_transformers,
                self.cv_transformers,
                self.loss_transformers,
                self.matrix_transformers,
                self.graph_transformers
            ]
        elif target_model_type == "rnn":
            stages = [
                self.basic_transformers,
                self.rnn_transformers,
                self.activation_norm_transformers,
                self.embedding_transformers,
                self.loss_transformers,
                self.matrix_transformers,
                self.graph_transformers
            ]
        else:  # mixed model
            stages = [
                self.basic_transformers,
                self.conv_transformers,
                self.activation_norm_transformers,
                self.attention_transformers,
                self.rnn_transformers,
                self.embedding_transformers,
                self.cv_transformers,
                self.loss_transformers,
                self.matrix_transformers,
                self.graph_transformers
            ]

        # Apply optimization stages
        while modified and iteration < max_iterations:
            modified = False

            for stage in stages:
                stage_modified = False
                for transformer in stage:
                    if transformer.apply(graph):
                        stage_modified = True
                        modified = True

                # Log stage completion
                if stage_modified:
                    print(f"Stage {stages.index(stage)} modified graph in iteration {iteration}")

            iteration += 1

        print(f"Optimization completed after {iteration} iterations")
        return graph


# Practical usage examples for different model types
class ModelOptimizationExamples:

    @staticmethod
    def optimize_bert_model():
        """Example: Optimizing BERT-like transformer model"""
        import onnx
        from onnxruntime.transformers.onnx_model_bert import BertOnnxModel

        # Load BERT model
        model_path = "bert-base-uncased.onnx"
        bert_model = BertOnnxModel(onnx.load(model_path))

        # Initialize optimization pipeline
        optimizer = ComprehensiveOptimizationPipeline()

        # Apply transformer-specific optimizations
        optimized_graph = optimizer.optimize_graph(
            bert_model.graph,
            target_model_type="transformer"
        )

        # Additional BERT-specific optimizations
        bert_model.optimize()  # Built-in BERT optimizations

        # Save optimized model
        bert_model.save_model_to_file("bert-optimized.onnx")

        return bert_model

    @staticmethod
    def optimize_resnet_model():
        """Example: Optimizing ResNet-like CNN model"""
        import onnx

        # Load ResNet model
        model = onnx.load("resnet50.onnx")

        # Initialize optimization pipeline
        optimizer = ComprehensiveOptimizationPipeline()

        # Apply CNN-specific optimizations
        optimized_graph = optimizer.optimize_graph(
            model.graph,
            target_model_type="cnn"
        )

        # Additional manual optimizations for ResNet
        from onnxruntime.capi._pybind_state import apply_optimization_level

        # Apply aggressive optimization
        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Create optimized session
        session = onnxruntime.InferenceSession(
            "resnet50.onnx",
            sess_options=sess_options
        )

        return session

    @staticmethod
    def optimize_gpt_model():
        """Example: Optimizing GPT-like decoder model"""
        import onnx
        from onnxruntime.transformers.onnx_model_gpt2 import Gpt2OnnxModel

        # Load GPT-2 model
        model_path = "gpt2-lm-head.onnx"
        gpt_model = Gpt2OnnxModel(onnx.load(model_path))

        # Initialize optimization pipeline
        optimizer = ComprehensiveOptimizationPipeline()

        # Apply transformer optimizations with decoder focus
        optimized_graph = optimizer.optimize_graph(
            gpt_model.graph,
            target_model_type="transformer"
        )

        # GPT-specific optimizations
        gpt_model.optimize()

        # Save optimized model
        gpt_model.save_model_to_file("gpt2-optimized.onnx")

        return gpt_model

    @staticmethod
    def optimize_mixed_model():
        """Example: Optimizing model with CNN + Transformer components"""
        import onnx

        # Load mixed model (e.g., Vision Transformer)
        model = onnx.load("vision_transformer.onnx")

        # Initialize optimization pipeline
        optimizer = ComprehensiveOptimizationPipeline()

        # Apply comprehensive optimizations
        optimized_graph = optimizer.optimize_graph(
            model.graph,
            target_model_type="mixed"
        )

        # Save optimized model
        onnx.save(model, "vision_transformer_optimized.onnx")

        return model


# Advanced optimization patterns detection
class AdvancedPatternDetector:
    """Detect complex optimization opportunities"""

    @staticmethod
    def detect_transformer_block_pattern(graph):
        """Detect complete transformer block for end-to-end optimization"""
        transformer_blocks = []

        for node in graph.nodes:
            if node.op_type == "LayerNormalization":
                # Look for transformer block pattern:
                # LayerNorm → MultiHeadAttention → Add → LayerNorm → FFN → Add
                block_pattern = TransformerBlockPattern()

                if block_pattern.analyze_from_layernorm(node, graph):
                    transformer_blocks.append(block_pattern)

        return transformer_blocks

    @staticmethod
    def detect_residual_block_pattern(graph):
        """Detect ResNet-style residual blocks"""
        residual_blocks = []

        for node in graph.nodes:
            if node.op_type == "Add":
                # Look for residual pattern:
                # Input → Conv → BN → ReLU → Conv → BN → Add(+Input) → ReLU
                residual_pattern = ResidualBlockPattern()

                if residual_pattern.analyze_from_add(node, graph):
                    residual_blocks.append(residual_pattern)

        return residual_blocks

    @staticmethod
    def detect_bottleneck_pattern(graph):
        """Detect bottleneck patterns for optimization"""
        bottlenecks = []

        # Analyze computational intensity vs memory bandwidth
        for node in graph.nodes:
            if node.op_type in ["MatMul", "Conv", "Gemm"]:
                intensity = calculate_computational_intensity(node)
                if intensity < BOTTLENECK_THRESHOLD:
                    bottlenecks.append({
                        'node': node,
                        'type': 'memory_bound',
                        'intensity': intensity
                    })

        return bottlenecks


# Production optimization pipeline
class ProductionOptimizer:
    """Production-ready optimization with performance monitoring"""

    def __init__(self, target_latency_ms=None, target_throughput=None):
        self.target_latency_ms = target_latency_ms
        self.target_throughput = target_throughput
        self.optimization_history = []

    def optimize_with_constraints(self, model_path, output_path):
        """Optimize model while respecting performance constraints"""
        import onnxruntime
        import time
        import numpy as np

        # Load original model
        original_model = onnx.load(model_path)
        baseline_perf = self.benchmark_model(model_path)

        # Initialize optimization pipeline
        optimizer = ComprehensiveOptimizationPipeline()

        # Progressive optimization with validation
        optimization_levels = [
            ("basic", ["basic_transformers"]),
            ("fusion", ["basic_transformers", "conv_transformers", "activation_norm_transformers"]),
            ("advanced", ["basic_transformers", "conv_transformers", "activation_norm_transformers",
                         "attention_transformers", "matrix_transformers"]),
            ("aggressive", ["basic_transformers", "conv_transformers", "activation_norm_transformers",
                           "attention_transformers", "matrix_transformers", "graph_transformers"])
        ]

        best_model = None
        best_perf = baseline_perf

        for level_name, transformer_groups in optimization_levels:
            # Apply optimization level
            optimized_model = self.apply_optimization_level(
                original_model, optimizer, transformer_groups
            )

            # Save temporary model
            temp_path = f"temp_{level_name}.onnx"
            onnx.save(optimized_model, temp_path)

            # Benchmark optimized model
            perf = self.benchmark_model(temp_path)

            # Check if optimization meets constraints
            if self.meets_constraints(perf, baseline_perf):
                best_model = optimized_model
                best_perf = perf

                self.optimization_history.append({
                    'level': level_name,
                    'performance': perf,
                    'improvement': self.calculate_improvement(baseline_perf, perf)
                })
            else:
                print(f"Optimization level {level_name} violates constraints")
                break

        # Save best model
        if best_model:
            onnx.save(best_model, output_path)
            print(f"Optimization completed. Final improvement: {self.calculate_improvement(baseline_perf, best_perf)}")
        else:
            print("No optimization met the constraints")

        return best_model, self.optimization_history

    def benchmark_model(self, model_path, num_runs=100):
        """Benchmark model performance"""
        import onnxruntime
        import time
        import numpy as np

        # Create session
        session = onnxruntime.InferenceSession(model_path)

        # Get input info
        input_info = session.get_inputs()[0]
        input_shape = input_info.shape

        # Handle dynamic shapes
        batch_size = 1 if isinstance(input_shape[0], str) else input_shape[0]
        if isinstance(input_shape[0], str):
            input_shape = [batch_size] + list(input_shape[1:])

        # Create dummy input
        dummy_input = np.random.randn(*input_shape).astype(np.float32)

        # Warmup
        for _ in range(10):
            _ = session.run(None, {input_info.name: dummy_input})

        # Measure latency
        latencies = []
        for _ in range(num_runs):
            start_time = time.time()
            _ = session.run(None, {input_info.name: dummy_input})
            latencies.append((time.time() - start_time) * 1000)  # Convert to ms

        return {
            'avg_latency_ms': np.mean(latencies),
            'p95_latency_ms': np.percentile(latencies, 95),
            'throughput_qps': 1000 / np.mean(latencies)
        }

    def meets_constraints(self, current_perf, baseline_perf):
        """Check if performance meets constraints"""
        if self.target_latency_ms:
            if current_perf['avg_latency_ms'] > self.target_latency_ms:
                return False

        if self.target_throughput:
            if current_perf['throughput_qps'] < self.target_throughput:
                return False

        # Ensure no regression beyond 5%
        latency_regression = (current_perf['avg_latency_ms'] - baseline_perf['avg_latency_ms']) / baseline_perf['avg_latency_ms']
        if latency_regression > 0.05:
            return False

        return True

    def calculate_improvement(self, baseline, current):
        """Calculate performance improvement"""
        latency_improvement = (baseline['avg_latency_ms'] - current['avg_latency_ms']) / baseline['avg_latency_ms'] * 100
        throughput_improvement = (current['throughput_qps'] - baseline['throughput_qps']) / baseline['throughput_qps'] * 100

        return {
            'latency_improvement_pct': latency_improvement,
            'throughput_improvement_pct': throughput_improvement
        }


# Example usage
if __name__ == "__main__":
    # Example 1: Basic optimization
    examples = ModelOptimizationExamples()
    optimized_bert = examples.optimize_bert_model()

    # Example 2: Production optimization with constraints
    prod_optimizer = ProductionOptimizer(
        target_latency_ms=50,  # Must be under 50ms
        target_throughput=20   # Must achieve 20 QPS
    )

    optimized_model, history = prod_optimizer.optimize_with_constraints(
        "model.onnx",
        "model_optimized.onnx"
    )

    print("Optimization history:", history)
```

#### Real-World Graph Optimization Examples

#### Example 1: ResNet Block Optimization
```python
import onnx
import onnxruntime as ort
import numpy as np

def optimize_resnet_block():
    """Example: Optimize a ResNet residual block with graph transformations"""

    # Create session with maximum optimization
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.optimized_model_filepath = "resnet_optimized.onnx"

    session = ort.InferenceSession("resnet_block.onnx", options)

    # Test performance before and after optimization
    input_data = np.random.randn(1, 256, 56, 56).astype(np.float32)

    # Warm up
    for _ in range(10):
        session.run(None, {"input": input_data})

    # Benchmark optimized model
    import time
    start_time = time.time()
    for _ in range(100):
        outputs = session.run(None, {"input": input_data})
    end_time = time.time()

    avg_time = (end_time - start_time) / 100
    print(f"Average inference time: {avg_time*1000:.2f} ms")

    return outputs

# Example: Manual graph transformation for specific pattern
def manual_conv_bn_fusion_example():
    """Manually fuse Conv + BatchNorm pattern"""

    # Load original model
    model = onnx.load("conv_bn_model.onnx")
    graph = model.graph

    # Find Conv → BatchNorm pattern
    conv_nodes = [n for n in graph.node if n.op_type == "Conv"]

    for conv_node in conv_nodes:
        # Find BatchNorm node that follows this Conv
        conv_output = conv_node.output[0]
        bn_node = None

        for node in graph.node:
            if node.op_type == "BatchNormalization" and conv_output in node.input:
                bn_node = node
                break

        if bn_node:
            # Get BatchNorm parameters
            bn_scale = get_initializer_value(graph, bn_node.input[1])
            bn_bias = get_initializer_value(graph, bn_node.input[2])
            bn_mean = get_initializer_value(graph, bn_node.input[3])
            bn_var = get_initializer_value(graph, bn_node.input[4])
            epsilon = get_attribute_value(bn_node, "epsilon", 1e-5)

            # Get Conv weights and bias
            conv_weights = get_initializer_value(graph, conv_node.input[1])
            conv_bias = get_initializer_value(graph, conv_node.input[2]) if len(conv_node.input) > 2 else np.zeros(conv_weights.shape[0])

            # Fuse BatchNorm into Conv
            # new_weight = weight * (scale / sqrt(var + epsilon))
            # new_bias = (bias - mean) * (scale / sqrt(var + epsilon)) + bn_bias

            scale_factor = bn_scale / np.sqrt(bn_var + epsilon)

            # Reshape scale_factor for broadcasting
            if len(conv_weights.shape) == 4:  # 2D Conv
                scale_factor = scale_factor.reshape(-1, 1, 1, 1)
            elif len(conv_weights.shape) == 3:  # 1D Conv
                scale_factor = scale_factor.reshape(-1, 1, 1)

            fused_weights = conv_weights * scale_factor
            fused_bias = (conv_bias - bn_mean) * bn_scale / np.sqrt(bn_var + epsilon) + bn_bias

            # Update Conv node with fused parameters
            update_initializer(graph, conv_node.input[1], fused_weights)
            if len(conv_node.input) > 2:
                update_initializer(graph, conv_node.input[2], fused_bias)
            else:
                # Add bias if it didn't exist
                bias_name = conv_node.name + "_fused_bias"
                add_initializer(graph, bias_name, fused_bias)
                conv_node.input.append(bias_name)

            # Redirect BatchNorm output to Conv output
            bn_output = bn_node.output[0]
            conv_node.output[0] = bn_output

            # Remove BatchNorm node
            graph.node.remove(bn_node)

            print(f"Fused Conv {conv_node.name} with BatchNorm {bn_node.name}")

    # Save optimized model
    onnx.save(model, "conv_bn_fused.onnx")

def get_initializer_value(graph, name):
    """Helper function to get initializer tensor value"""
    for init in graph.initializer:
        if init.name == name:
            return onnx.numpy_helper.to_array(init)
    return None

def update_initializer(graph, name, new_value):
    """Helper function to update initializer tensor"""
    for i, init in enumerate(graph.initializer):
        if init.name == name:
            graph.initializer[i].CopyFrom(onnx.numpy_helper.from_array(new_value, name))
            break
```

#### Example 2: Transformer Optimization
```python
def optimize_transformer_attention():
    """Optimize transformer attention patterns"""

    # Enable advanced optimizations for transformers
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # Add transformer-specific optimizations
    options.add_session_config_entry("optimization.enable_transformer_layer_fusion", "1")
    options.add_session_config_entry("optimization.enable_attention_fusion", "1")
    options.add_session_config_entry("optimization.enable_skip_layer_norm_fusion", "1")

    session = ort.InferenceSession("transformer_model.onnx", options)

    # Test with typical transformer inputs
    batch_size, seq_len, hidden_size = 8, 512, 768
    input_ids = np.random.randint(0, 30000, (batch_size, seq_len))
    attention_mask = np.ones((batch_size, seq_len))

    inputs = {
        "input_ids": input_ids.astype(np.int64),
        "attention_mask": attention_mask.astype(np.int64)
    }

    # Benchmark performance
    times = []
    for _ in range(50):
        start = time.time()
        outputs = session.run(None, inputs)
        times.append(time.time() - start)

    avg_time = np.mean(times[10:])  # Skip warmup
    throughput = batch_size / avg_time

    print(f"Average inference time: {avg_time*1000:.2f} ms")
    print(f"Throughput: {throughput:.2f} samples/sec")

    return outputs

# Example: Custom attention fusion implementation
class CustomAttentionFusion:
    def __init__(self):
        self.fused_patterns = []

    def find_attention_pattern(self, graph):
        """Find multi-head attention patterns in the graph"""
        attention_patterns = []

        # Look for typical attention pattern:
        # Input → [Q,K,V projections] → QK^T → Scale → Softmax → Attention*V → Output

        for node in graph.node:
            if node.op_type == "MatMul":
                # Check if this could be Q*K^T
                pattern = self.analyze_qk_pattern(graph, node)
                if pattern:
                    attention_patterns.append(pattern)

        return attention_patterns

    def analyze_qk_pattern(self, graph, qk_node):
        """Analyze if a MatMul node is part of attention Q*K^T"""
        # Find the softmax that follows this MatMul
        qk_output = qk_node.output[0]

        # Look for: MatMul → (Add/Div for scaling) → Softmax → MatMul
        next_nodes = self.find_consumers(graph, qk_output)

        for next_node in next_nodes:
            if next_node.op_type in ["Add", "Div", "Mul"]:  # Scaling operation
                scale_output = next_node.output[0]
                softmax_nodes = self.find_consumers(graph, scale_output)

                for softmax_node in softmax_nodes:
                    if softmax_node.op_type == "Softmax":
                        # Found attention pattern
                        av_nodes = self.find_consumers(graph, softmax_node.output[0])

                        for av_node in av_nodes:
                            if av_node.op_type == "MatMul":
                                return {
                                    "qk_matmul": qk_node,
                                    "scale_op": next_node,
                                    "softmax": softmax_node,
                                    "av_matmul": av_node
                                }

        return None

    def find_consumers(self, graph, output_name):
        """Find all nodes that consume a given output"""
        consumers = []
        for node in graph.node:
            if output_name in node.input:
                consumers.append(node)
        return consumers

    def fuse_attention_pattern(self, graph, pattern):
        """Replace attention pattern with fused operation"""
        # Create fused attention node
        fused_node = onnx.helper.make_node(
            "MultiHeadAttention",
            inputs=[
                pattern["qk_matmul"].input[0],  # Query
                pattern["qk_matmul"].input[1],  # Key
                pattern["av_matmul"].input[1],  # Value
            ],
            outputs=pattern["av_matmul"].output,
            name="FusedAttention_" + pattern["qk_matmul"].name
        )

        # Add to graph
        graph.node.append(fused_node)

        # Remove original nodes
        nodes_to_remove = [
            pattern["qk_matmul"],
            pattern["scale_op"],
            pattern["softmax"],
            pattern["av_matmul"]
        ]

        for node in nodes_to_remove:
            if node in graph.node:
                graph.node.remove(node)

        print(f"Fused attention pattern: {fused_node.name}")
```### IO Binding for High Performance

```python
# Use IO binding to avoid memory copies
session = ort.InferenceSession("model.onnx")

# Bind input/output buffers
io_binding = session.io_binding()

# Bind input on GPU
input_ortvalue = ort.OrtValue.ortvalue_from_numpy(input_array, 'cuda', 0)
io_binding.bind_input('input', input_ortvalue)

# Bind output on GPU
output_ortvalue = ort.OrtValue.ortvalue_from_shape_and_type(
    output_shape, np.float32, 'cuda', 0)
io_binding.bind_output('output', output_ortvalue)

# Run with IO binding
session.run_with_iobinding(io_binding)

# Get output
output_array = output_ortvalue.numpy()
```

### Model Optimization Techniques

### Model Quantization: Comprehensive Theory and Implementation

Quantization is a fundamental technique for model compression and acceleration that represents weights and activations with lower precision data types, reducing model size and improving inference speed while maintaining acceptable accuracy.

#### Theoretical Foundations

#### 1. Quantization Mathematics

**Uniform Quantization (Linear)**:
```
Q(x) = clamp(round((x - zero_point) / scale), qmin, qmax)
DQ(q) = scale * (q - zero_point)
```

**Non-Uniform Quantization (Logarithmic)**:
```
Q(x) = sign(x) * round(log2(|x|/α) / β)
DQ(q) = sign(q) * α * 2^(β * |q|)
```

Where:
- **Scale (s)**: `s = (xmax - xmin) / (qmax - qmin)`
- **Zero Point (z)**: `z = qmin - round(xmin / s)`
- **Quantization Range**: `[qmin, qmax]` (e.g., [0, 255] for uint8)

#### 2. Precision Types and Characteristics

```python
# Comprehensive precision type definitions
class PrecisionTypes:
    """All supported quantization precision types"""

    # Integer Types
    INT8 = {
        'range': [-128, 127],
        'bits': 8,
        'signed': True,
        'memory_saving': '4x vs FP32',
        'compute_efficiency': 'High on CPU/GPU'
    }

    UINT8 = {
        'range': [0, 255],
        'bits': 8,
        'signed': False,
        'memory_saving': '4x vs FP32',
        'compute_efficiency': 'High on CPU/GPU'
    }

    INT16 = {
        'range': [-32768, 32767],
        'bits': 16,
        'signed': True,
        'memory_saving': '2x vs FP32',
        'compute_efficiency': 'Medium'
    }

    # Floating Point Types
    FP16 = {
        'bits': 16,
        'format': 'IEEE 754 half-precision',
        'memory_saving': '2x vs FP32',
        'compute_efficiency': 'High on modern GPUs',
        'mantissa': 10,
        'exponent': 5
    }

    BF16 = {
        'bits': 16,
        'format': 'Brain Floating Point',
        'memory_saving': '2x vs FP32',
        'dynamic_range': 'Same as FP32',
        'mantissa': 7,
        'exponent': 8
    }

    # Custom Types
    INT4 = {
        'range': [-8, 7],
        'bits': 4,
        'memory_saving': '8x vs FP32',
        'use_cases': 'Extreme compression'
    }

    # Binary/Ternary
    BINARY = {
        'values': [-1, 1],
        'bits': 1,
        'memory_saving': '32x vs FP32',
        'accuracy_trade_off': 'High'
    }

    TERNARY = {
        'values': [-1, 0, 1],
        'bits': 2,
        'memory_saving': '16x vs FP32',
        'accuracy_trade_off': 'Medium-High'
    }
```

#### 3. Quantization Algorithms

#### A. Post-Training Quantization (PTQ)

```python
from onnxruntime.quantization import quantize_dynamic, quantize_static, QuantType, QuantFormat
import numpy as np
import onnx

class PostTrainingQuantization:
    """Comprehensive PTQ implementation"""

    def __init__(self):
        self.supported_ops = [
            'Conv', 'MatMul', 'Gemm', 'Add', 'Mul', 'Relu', 'Sigmoid',
            'Tanh', 'BatchNormalization', 'LayerNormalization'
        ]

    def dynamic_quantization(self, model_path, output_path, precision='int8'):
        """
        Dynamic Quantization: Quantize weights offline, activations online

        Pros: No calibration data needed, fast setup
        Cons: Runtime overhead for activation quantization
        """
        precision_map = {
            'int8': QuantType.QInt8,
            'uint8': QuantType.QUInt8,
            'int16': QuantType.QInt16,
            'uint16': QuantType.QUInt16
        }

        # Basic dynamic quantization
        quantize_dynamic(
            model_input=model_path,
            model_output=output_path,
            weight_type=precision_map[precision],
            per_channel=True,  # Better accuracy for Conv/Linear layers
            reduce_range=True,  # Avoid overflow on some hardware
            optimize_model=True,
            op_types_to_quantize=['Conv', 'MatMul', 'Attention'],
            extra_options={
                'EnableSubgraph': True,
                'ForceQuantizeNoInputCheck': False,
                'MatMulConstBOnly': True  # Only quantize constant B matrix
            }
        )

        return self._analyze_quantization_impact(model_path, output_path)

    def static_quantization_int8(self, model_path, output_path, calibration_data):
        """
        Static INT8 Quantization with comprehensive calibration

        Pros: Best performance, no runtime overhead
        Cons: Requires representative calibration data
        """

        # Create calibration data reader
        calibration_reader = self._create_calibration_reader(
            calibration_data, model_path)

        # Comprehensive static quantization
        quantize_static(
            model_input=model_path,
            model_output=output_path,
            calibration_data_reader=calibration_reader,
            quant_format=QuantFormat.QOperator,  # Use quantized operators
            per_channel=True,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QUInt8,
            optimize_model=True,
            use_external_data_format=True,
            calibrate_method='MinMax',  # or 'Entropy', 'Percentile'
            extra_options={
                'ActivationSymmetric': False,  # Asymmetric for better range
                'WeightSymmetric': True,       # Symmetric for weights
                'EnableSubgraph': True,
                'ForceQuantizeNoInputCheck': False,
                'DedicatedQDQPair': False,
                'QDQOpTypePerChannelSupportToAxis': {
                    'MatMul': {'B': 0},
                    'Conv': {'weight': 0}
                }
            }
        )

        return self._validate_quantized_model(model_path, output_path, calibration_data)

    def qdq_quantization(self, model_path, output_path, calibration_data):
        """
        QDQ (Quantize-Dequantize) Format for maximum flexibility

        Pros: Explicit Q/DQ ops, hardware-agnostic, fine-grained control
        Cons: Larger model size due to explicit ops
        """
        calibration_reader = self._create_calibration_reader(
            calibration_data, model_path)

        quantize_static(
            model_input=model_path,
            model_output=output_path,
            calibration_data_reader=calibration_reader,
            quant_format=QuantFormat.QDQ,
            per_channel=True,
            reduce_range=False,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QUInt8,
            optimize_model=True,
            # Selective quantization
            nodes_to_quantize=self._select_quantizable_nodes(model_path),
            nodes_to_exclude=self._get_sensitive_nodes(model_path),
            op_types_to_quantize=['Conv', 'MatMul', 'Gemm', 'Add'],
            extra_options={
                'MinimalBuild': False,
                'EnableSubgraph': True,
                'ForceQuantizeNoInputCheck': True
            }
        )

        return self._optimize_qdq_model(output_path)

    def mixed_precision_quantization(self, model_path, output_path, precision_map):
        """
        Mixed-precision quantization: Different precisions for different layers

        Args:
            precision_map: Dict mapping layer names to precision types
                e.g., {'conv1': 'int8', 'attention': 'int16', 'output': 'fp16'}
        """
        model = onnx.load(model_path)

        # Group nodes by precision requirements
        precision_groups = {}
        for node in model.graph.node:
            if node.name in precision_map:
                precision = precision_map[node.name]
                if precision not in precision_groups:
                    precision_groups[precision] = []
                precision_groups[precision].append(node.name)

        # Apply quantization per precision group
        current_model = model_path
        for precision, node_names in precision_groups.items():
            temp_output = f"temp_{precision}_{output_path}"

            if precision == 'int8':
                self._quantize_nodes_int8(current_model, temp_output, node_names)
            elif precision == 'int16':
                self._quantize_nodes_int16(current_model, temp_output, node_names)
            elif precision == 'fp16':
                self._quantize_nodes_fp16(current_model, temp_output, node_names)

            current_model = temp_output

        # Final optimization pass
        self._optimize_mixed_precision_model(current_model, output_path)

        return self._benchmark_mixed_precision(model_path, output_path)
```

#### B. Quantization-Aware Training (QAT)

```python
import torch
import torch.nn as nn
from torch.quantization import QuantStub, DeQuantStub, prepare_qat, convert

class QuantizationAwareTraining:
    """Comprehensive QAT implementation for training-time quantization"""

    def __init__(self):
        self.qconfig_mapping = {
            'int8_qat': {
                'weight': torch.quantization.default_weight_observer,
                'activation': torch.quantization.default_observer
            },
            'int4_qat': {
                'weight': torch.quantization.PerChannelMinMaxObserver.with_args(
                    dtype=torch.qint8, qscheme=torch.per_channel_symmetric),
                'activation': torch.quantization.MinMaxObserver.with_args(
                    dtype=torch.quint8, qscheme=torch.per_tensor_affine)
            }
        }

    def prepare_model_for_qat(self, model, precision='int8'):
        """Prepare PyTorch model for quantization-aware training"""

        # Add quantization stubs
        class QATModel(nn.Module):
            def __init__(self, original_model):
                super().__init__()
                self.quant = QuantStub()
                self.model = original_model
                self.dequant = DeQuantStub()

            def forward(self, x):
                x = self.quant(x)
                x = self.model(x)
                x = self.dequant(x)
                return x

        qat_model = QATModel(model)

        # Configure quantization
        qat_model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')

        # Prepare for QAT
        qat_model_prepared = prepare_qat(
            qat_model,
            inplace=False,
            observer_non_leaf_module_list=[]
        )

        return qat_model_prepared

    def train_with_quantization(self, model, train_loader, val_loader, epochs=10):
        """Train model with quantization awareness"""

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

        # Training loop with quantization
        for epoch in range(epochs):
            model.train()

            # Enable fake quantization during training
            model.apply(torch.quantization.enable_fake_quant)
            model.apply(torch.quantization.enable_observer)

            train_loss = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                # Update quantization parameters
                if batch_idx % 100 == 0:
                    self._update_quantization_ranges(model)

            # Validation with quantization
            val_accuracy = self._validate_qat_model(model, val_loader)

            print(f'Epoch {epoch}: Train Loss: {train_loss/len(train_loader):.4f}, '
                  f'Val Accuracy: {val_accuracy:.4f}')

            scheduler.step()

            # Gradually reduce quantization noise
            if epoch > 5:
                self._reduce_quantization_noise(model, epoch)

        return model

    def convert_qat_to_quantized(self, qat_model):
        """Convert QAT model to fully quantized model"""

        # Disable fake quantization and observers
        qat_model.eval()
        qat_model.apply(torch.quantization.disable_fake_quant)
        qat_model.apply(torch.quantization.disable_observer)

        # Convert to quantized model
        quantized_model = convert(qat_model, inplace=False)

        return quantized_model

    def export_qat_to_onnx(self, qat_model, dummy_input, output_path):
        """Export QAT model to ONNX with quantization info"""

        # Convert to quantized model first
        quantized_model = self.convert_qat_to_quantized(qat_model)

        # Export to ONNX
        torch.onnx.export(
            quantized_model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )

        return output_path

    def progressive_quantization_training(self, model, train_loader, val_loader):
        """Progressive quantization: gradually increase quantization during training"""

        # Phase 1: Full precision training
        print("Phase 1: Full precision training...")
        model = self._train_full_precision(model, train_loader, val_loader, epochs=5)

        # Phase 2: 16-bit quantization
        print("Phase 2: 16-bit quantization...")
        model_16bit = self._apply_16bit_quantization(model)
        model_16bit = self._fine_tune_quantized(model_16bit, train_loader, val_loader, epochs=3)

        # Phase 3: 8-bit quantization
        print("Phase 3: 8-bit quantization...")
        model_8bit = self._apply_8bit_quantization(model_16bit)
        model_8bit = self._fine_tune_quantized(model_8bit, train_loader, val_loader, epochs=3)

        # Phase 4: Mixed precision optimization
        print("Phase 4: Mixed precision optimization...")
        model_optimized = self._optimize_mixed_precision(model_8bit, train_loader, val_loader)

        return model_optimized
```

#### C. Advanced Quantization Techniques

```python
class AdvancedQuantizationTechniques:
    """State-of-the-art quantization methods"""

    def knowledge_distillation_quantization(self, teacher_model, student_model_path,
                                          train_loader, temperature=4.0, alpha=0.7):
        """
        Knowledge Distillation for Quantization
        Teacher: Full precision model
        Student: Quantized model
        """
        import torch.nn.functional as F

        # Load teacher model
        teacher = torch.load(teacher_model)
        teacher.eval()

        # Prepare student model for quantization
        student = self.prepare_model_for_qat(torch.load(student_model_path))

        optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)
        kl_loss = nn.KLDivLoss(reduction='batchmean')
        ce_loss = nn.CrossEntropyLoss()

        for epoch in range(10):
            student.train()

            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()

                # Forward pass through both models
                with torch.no_grad():
                    teacher_output = teacher(data)
                    teacher_soft = F.softmax(teacher_output / temperature, dim=1)

                student_output = student(data)
                student_soft = F.log_softmax(student_output / temperature, dim=1)

                # Combined loss: knowledge distillation + task loss
                distill_loss = kl_loss(student_soft, teacher_soft) * (temperature ** 2)
                task_loss = ce_loss(student_output, target)

                total_loss = alpha * distill_loss + (1 - alpha) * task_loss

                total_loss.backward()
                optimizer.step()

        return student

    def sensitivity_aware_quantization(self, model_path, test_loader):
        """
        Analyze layer sensitivity and apply adaptive quantization
        """
        import onnx
        from onnxruntime.quantization.calibrate import create_calibrator

        model = onnx.load(model_path)
        sensitivity_scores = {}

        # Test each layer individually
        for node in model.graph.node:
            if node.op_type in ['Conv', 'MatMul', 'Gemm']:
                # Create model with only this node quantized
                temp_model = self._quantize_single_node(model, node)

                # Measure accuracy drop
                accuracy_drop = self._measure_accuracy_drop(
                    model_path, temp_model, test_loader)

                sensitivity_scores[node.name] = accuracy_drop

        # Adaptive quantization based on sensitivity
        return self._apply_adaptive_quantization(model_path, sensitivity_scores)

    def bit_width_search(self, model_path, target_size_mb, accuracy_threshold=0.95):
        """
        Automated bit-width search to meet size constraints
        """
        bit_widths = [1, 2, 4, 8, 16]
        layer_configs = []

        model = onnx.load(model_path)
        baseline_accuracy = self._measure_model_accuracy(model_path)

        for node in model.graph.node:
            if self._is_quantizable(node):
                best_config = None

                for bits in bit_widths:
                    # Test this bit width for this layer
                    temp_model = self._set_layer_bitwidth(model, node, bits)

                    model_size = self._calculate_model_size(temp_model)
                    accuracy = self._measure_model_accuracy(temp_model)

                    accuracy_ratio = accuracy / baseline_accuracy

                    if (accuracy_ratio >= accuracy_threshold and
                        model_size <= target_size_mb):
                        best_config = {
                            'node': node.name,
                            'bits': bits,
                            'accuracy': accuracy,
                            'size': model_size
                        }
                        break

                if best_config:
                    layer_configs.append(best_config)

        # Apply optimal bit-width configuration
        return self._apply_bitwidth_config(model_path, layer_configs)

    def outlier_aware_quantization(self, model_path, outlier_threshold=3.0):
        """
        Handle outliers in weight/activation distributions
        """
        model = onnx.load(model_path)

        for node in model.graph.node:
            if self._has_weights(node):
                weights = self._extract_weights(node)

                # Detect outliers using statistical methods
                outliers = self._detect_outliers(weights, outlier_threshold)

                if len(outliers) > 0:
                    # Apply outlier-aware quantization
                    self._apply_outlier_quantization(node, weights, outliers)

        return self._rebuild_model_with_outlier_handling(model)

    def channel_wise_quantization(self, model_path, granularity='per_channel'):
        """
        Fine-grained quantization at different granularities
        """
        granularity_options = {
            'per_tensor': self._quantize_per_tensor,
            'per_channel': self._quantize_per_channel,
            'per_group': self._quantize_per_group,
            'per_block': self._quantize_per_block
        }

        quantize_func = granularity_options[granularity]
        return quantize_func(model_path)
```

#### D. Hardware-Specific Quantization

```python
class HardwareSpecificQuantization:
    """Quantization optimized for specific hardware targets"""

    def cpu_quantization_x86(self, model_path, output_path):
        """Optimized for Intel x86 CPUs with VNNI support"""

        quantize_static(
            model_input=model_path,
            model_output=output_path,
            calibration_data_reader=self.calibration_reader,
            quant_format=QuantFormat.QOperator,
            per_channel=True,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QUInt8,
            # x86-specific optimizations
            extra_options={
                'UseVNNI': True,  # Vector Neural Network Instructions
                'EnableSubgraph': True,
                'OptimizeForIntel': True,
                'UseAVX512': True
            }
        )

    def gpu_quantization_cuda(self, model_path, output_path):
        """Optimized for NVIDIA GPUs with Tensor Cores"""

        # Use INT8 for Tensor Core acceleration
        quantize_static(
            model_input=model_path,
            model_output=output_path,
            calibration_data_reader=self.calibration_reader,
            quant_format=QuantFormat.QDQ,  # Better for GPU
            per_channel=True,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QInt8,  # Symmetric for GPU
            extra_options={
                'EnableCUDAGraphOptimization': True,
                'UseTensorCores': True,
                'OptimizeForTensorRT': True,
                'ActivationSymmetric': True  # Better for GPU
            }
        )

    def mobile_quantization_arm(self, model_path, output_path):
        """Optimized for ARM processors (mobile devices)"""

        quantize_dynamic(
            model_input=model_path,
            model_output=output_path,
            weight_type=QuantType.QUInt8,  # Unsigned better for ARM
            per_channel=False,  # Simpler for mobile
            reduce_range=True,  # Avoid overflow
            extra_options={
                'OptimizeForARM': True,
                'ReduceMemoryFootprint': True,
                'EnableNEON': True,  # ARM NEON SIMD
                'MinimalBuild': True
            }
        )

    def edge_quantization_extreme(self, model_path, output_path):
        """Extreme quantization for edge devices (4-bit, binary)"""

        # 4-bit quantization
        self._apply_4bit_quantization(model_path, f"{output_path}_4bit.onnx")

        # Binary quantization for ultra-low power
        self._apply_binary_quantization(model_path, f"{output_path}_binary.onnx")

        # Ternary quantization (compromise)
        self._apply_ternary_quantization(model_path, f"{output_path}_ternary.onnx")
```

#### E. Floating Point Quantization (FP16/BF16)

```python
class FloatingPointQuantization:
    """Half-precision floating point quantization"""

    def fp16_quantization(self, model_path, output_path):
        """Convert model to FP16 (IEEE 754 half-precision)"""
        import onnx
        from onnxmltools.utils.float16_converter import convert_float_to_float16

        # Load model
        model = onnx.load(model_path)

        # Convert to FP16
        model_fp16 = convert_float_to_float16(
            model,
            min_positive_val=1e-7,
            max_finite_val=1e4,
            keep_io_types=True,  # Keep input/output as FP32
            disable_shape_infer=False,
            op_block_list=['Softmax', 'Sigmoid', 'Tanh'],  # Keep sensitive ops in FP32
            node_block_list=[]
        )

        # Save FP16 model
        onnx.save(model_fp16, output_path)

        return self._validate_fp16_model(model_path, output_path)

    def bf16_quantization(self, model_path, output_path):
        """Convert model to BF16 (Brain Floating Point)"""

        # BF16 has same dynamic range as FP32 but less precision
        model = onnx.load(model_path)

        # Convert specific nodes to BF16
        for node in model.graph.node:
            if node.op_type in ['Conv', 'MatMul', 'Gemm']:
                self._convert_node_to_bf16(node)

        onnx.save(model, output_path)

        return self._benchmark_bf16_performance(model_path, output_path)

    def mixed_fp16_fp32(self, model_path, output_path, fp32_ops=None):
        """Mixed precision: FP16 for most ops, FP32 for sensitive ops"""

        if fp32_ops is None:
            fp32_ops = ['Softmax', 'Sigmoid', 'LayerNormalization', 'BatchNormalization']

        model = onnx.load(model_path)

        for node in model.graph.node:
            if node.op_type not in fp32_ops:
                # Convert to FP16
                self._convert_node_to_fp16(node)
            # else: keep in FP32

        onnx.save(model, output_path)

        return {
            'model_size_reduction': self._calculate_size_reduction(model_path, output_path),
            'speed_improvement': self._measure_speed_improvement(model_path, output_path),
            'accuracy_preservation': self._measure_accuracy_preservation(model_path, output_path)
        }
```

#### F. Production Quantization Pipeline

```python
class ProductionQuantizationPipeline:
    """End-to-end quantization pipeline for production deployment"""

    def __init__(self, target_hardware='cpu', performance_target=None):
        self.target_hardware = target_hardware
        self.performance_target = performance_target
        self.quantization_strategies = {
            'cpu': self._cpu_quantization_strategy,
            'gpu': self._gpu_quantization_strategy,
            'mobile': self._mobile_quantization_strategy,
            'edge': self._edge_quantization_strategy
        }

    def auto_quantize(self, model_path, calibration_data, validation_data):
        """
        Automatic quantization with strategy selection based on model analysis
        """
        # Step 1: Analyze model characteristics
        model_analysis = self._analyze_model(model_path)

        # Step 2: Select quantization strategy
        strategy = self._select_strategy(model_analysis)

        # Step 3: Apply quantization
        quantized_models = {}

        for precision in strategy['precisions']:
            output_path = f"model_{precision}_{self.target_hardware}.onnx"

            if precision == 'int8':
                self._apply_int8_quantization(
                    model_path, output_path, calibration_data, strategy['int8_config'])
            elif precision == 'fp16':
                self._apply_fp16_quantization(
                    model_path, output_path, strategy['fp16_config'])
            elif precision == 'mixed':
                self._apply_mixed_precision(
                    model_path, output_path, strategy['mixed_config'])

            # Validate quantized model
            validation_results = self._validate_quantized_model(
                output_path, validation_data)

            quantized_models[precision] = {
                'path': output_path,
                'validation': validation_results,
                'meets_targets': self._check_performance_targets(validation_results)
            }

        # Step 4: Select best model
        best_model = self._select_best_model(quantized_models)

        # Step 5: Final optimization
        optimized_model = self._final_optimization(best_model['path'])

        return {
            'best_model': optimized_model,
            'all_models': quantized_models,
            'strategy_used': strategy,
            'optimization_report': self._generate_optimization_report(
                model_path, optimized_model, validation_data)
        }

    def benchmark_all_precisions(self, model_path, test_data):
        """Comprehensive benchmarking across all precision types"""

        precisions = ['fp32', 'fp16', 'bf16', 'int8', 'int4', 'mixed_int8_fp16']
        results = {}

        # Baseline FP32
        baseline_metrics = self._benchmark_model(model_path, test_data)
        results['fp32'] = baseline_metrics

        for precision in precisions[1:]:  # Skip fp32 (already done)
            try:
                # Apply quantization
                quantized_path = f"temp_{precision}.onnx"
                self._apply_precision(model_path, quantized_path, precision)

                # Benchmark
                metrics = self._benchmark_model(quantized_path, test_data)

                # Calculate improvements/degradations
                metrics['speedup'] = baseline_metrics['latency'] / metrics['latency']
                metrics['size_reduction'] = baseline_metrics['model_size'] / metrics['model_size']
                metrics['accuracy_drop'] = baseline_metrics['accuracy'] - metrics['accuracy']

                results[precision] = metrics

            except Exception as e:
                results[precision] = {'error': str(e)}

        return self._generate_precision_comparison_report(results)

    def deployment_ready_quantization(self, model_path, deployment_config):
        """
        Prepare model for specific deployment scenario

        deployment_config: {
            'target_latency_ms': 50,
            'max_model_size_mb': 100,
            'min_accuracy': 0.95,
            'batch_size': 1,
            'hardware': 'cpu'  # or 'gpu', 'mobile', 'edge'
        }
        """

        # Progressive quantization with validation
        quantization_levels = [
            ('fp16', 'Fast FP16 conversion'),
            ('int8_dynamic', 'Dynamic INT8 quantization'),
            ('int8_static', 'Static INT8 quantization'),
            ('mixed_precision', 'Mixed precision optimization'),
            ('aggressive', 'Aggressive compression')
        ]

        best_model = None

        for level, description in quantization_levels:
            print(f"Trying {description}...")

            try:
                quantized_path = f"model_{level}.onnx"
                self._apply_quantization_level(model_path, quantized_path, level)

                # Validate against deployment constraints
                validation = self._validate_deployment_constraints(
                    quantized_path, deployment_config)

                if validation['meets_all_constraints']:
                    best_model = {
                        'path': quantized_path,
                        'level': level,
                        'metrics': validation['metrics']
                    }
                    break
                else:
                    print(f"  - Failed: {validation['failed_constraints']}")

            except Exception as e:
                print(f"  - Error: {e}")
                continue

        if best_model:
            # Final optimization for deployment
            final_model = self._optimize_for_deployment(
                best_model['path'], deployment_config)

            return {
                'success': True,
                'model_path': final_model,
                'quantization_level': best_model['level'],
                'final_metrics': self._final_validation(final_model, deployment_config)
            }
        else:
            return {
                'success': False,
                'message': 'No quantization level met deployment constraints',
                'suggestions': self._suggest_constraint_relaxation(deployment_config)
            }


# Example usage and integration
class QuantizationExamples:
    """Real-world quantization examples"""

    @staticmethod
    def quantize_bert_for_production():
        """Example: BERT quantization for production inference"""

        ptq = PostTrainingQuantization()

        # Load calibration data
        calibration_data = load_bert_calibration_data(num_samples=1000)

        # Try different quantization approaches
        results = {}

        # 1. Dynamic INT8
        results['dynamic_int8'] = ptq.dynamic_quantization(
            "bert-base.onnx", "bert-dynamic-int8.onnx", precision='int8')

        # 2. Static INT8
        results['static_int8'] = ptq.static_quantization_int8(
            "bert-base.onnx", "bert-static-int8.onnx", calibration_data)

        # 3. Mixed precision
        precision_map = {
            'attention': 'int16',  # Keep attention in higher precision
            'layer_norm': 'fp16',  # Layer norm sensitive to quantization
            'embeddings': 'int8',  # Embeddings can be heavily quantized
            'feed_forward': 'int8'  # FFN can handle INT8 well
        }
        results['mixed_precision'] = ptq.mixed_precision_quantization(
            "bert-base.onnx", "bert-mixed.onnx", precision_map)

        return results

    @staticmethod
    def quantize_vision_model():
        """Example: ResNet/EfficientNet quantization"""

        pipeline = ProductionQuantizationPipeline(target_hardware='gpu')

        # Auto-quantization for vision model
        calibration_data = load_imagenet_samples(num_samples=500)
        validation_data = load_imagenet_validation()

        results = pipeline.auto_quantize(
            "efficientnet-b0.onnx",
            calibration_data,
            validation_data
        )

        return results

    @staticmethod
    def quantize_for_mobile_deployment():
        """Example: Mobile deployment quantization"""

        deployment_config = {
            'target_latency_ms': 100,   # 100ms max latency
            'max_model_size_mb': 50,    # 50MB max size
            'min_accuracy': 0.90,       # 90% min accuracy
            'batch_size': 1,            # Single inference
            'hardware': 'mobile'        # ARM processors
        }

        pipeline = ProductionQuantizationPipeline(target_hardware='mobile')

        result = pipeline.deployment_ready_quantization(
            "mobilenet-v3.onnx", deployment_config)

        if result['success']:
            print(f"Successfully quantized for mobile: {result['model_path']}")
            print(f"Final metrics: {result['final_metrics']}")
        else:
            print(f"Quantization failed: {result['message']}")
            print(f"Suggestions: {result['suggestions']}")

        return result
```
            quantized_outputs = quantized_session.run(None, test_data)

            # Calculate accuracy drop
            accuracy_drop = self._calculate_accuracy_drop(
                baseline_outputs, quantized_outputs)
            sensitivity_scores[node] = accuracy_drop

        return sensitivity_scores

    def selective_quantization(self, model_path, sensitivity_threshold=0.01):
        """Quantize only layers with low sensitivity"""
        sensitivity_scores = self.analyze_model_sensitivity(model_path, test_data)

        # Select nodes for quantization
        nodes_to_quantize = [
            node for node, score in sensitivity_scores.items()
            if score < sensitivity_threshold
        ]

        # Perform selective quantization
        quantize_static(
            model_input=model_path,
            model_output="model_selective_quantized.onnx",
            calibration_data_reader=calibration_data_reader,
            nodes_to_quantize=nodes_to_quantize,
            quant_format=QuantFormat.QDQ
        )
```

#### Advanced Quantization Techniques

#### Mixed-Bit Quantization
```python
def mixed_bit_quantization():
    """Use different bit widths for different layers"""
    # 8-bit for most layers, 16-bit for sensitive layers
    sensitive_layers = ['attention_output', 'layer_norm']

    # First pass: 8-bit quantization
    quantize_dynamic(
        model_input="model.onnx",
        model_output="model_8bit.onnx",
        weight_type=QuantType.QUInt8,
        nodes_to_exclude=sensitive_layers
    )

    # Second pass: 16-bit for sensitive layers
    quantize_dynamic(
        model_input="model_8bit.onnx",
        model_output="model_mixed_bit.onnx",
        weight_type=QuantType.QUInt16,
        nodes_to_quantize=sensitive_layers
    )
```

#### Post-Training Optimization
```python
class QuantizationOptimizer:
    def __init__(self):
        self.optimization_techniques = [
            'bias_correction',
            'cross_layer_equalization',
            'adaptive_rounding'
        ]

    def optimize_quantized_model(self, model_path, calibration_data):
        """Apply post-quantization optimizations"""

        # Bias correction: adjust biases to compensate for quantization errors
        self._apply_bias_correction(model_path, calibration_data)

        # Cross-layer equalization: balance weight ranges across layers
        self._apply_cross_layer_equalization(model_path)

        # Adaptive rounding: optimize rounding strategy
        self._apply_adaptive_rounding(model_path, calibration_data)

    def _apply_bias_correction(self, model_path, calibration_data):
        """Correct biases to compensate for quantization errors"""
        # Load original and quantized models
        original_session = ort.InferenceSession("original_model.onnx")
        quantized_session = ort.InferenceSession(model_path)

        for batch in calibration_data:
            # Get intermediate activations
            orig_activations = self._get_intermediate_outputs(
                original_session, batch)
            quant_activations = self._get_intermediate_outputs(
                quantized_session, batch)

            # Calculate bias corrections
            bias_corrections = self._calculate_bias_corrections(
                orig_activations, quant_activations)

            # Apply corrections to model
            self._update_model_biases(model_path, bias_corrections)
```

#### Real-World Quantization Examples

#### Example 1: ResNet50 Quantization Pipeline
```python
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, quantize_static, CalibrationDataReader
import numpy as np
from PIL import Image
import os

class ResNet50QuantizationPipeline:
    def __init__(self, model_path="resnet50.onnx"):
        self.model_path = model_path
        self.calibration_data_path = "imagenet_calibration/"

    def preprocess_image(self, image_path):
        """ImageNet preprocessing for ResNet50"""
        image = Image.open(image_path).convert('RGB')
        image = image.resize((224, 224))

        # Convert to numpy and normalize
        img_array = np.array(image).astype(np.float32) / 255.0

        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_array = (img_array - mean) / std

        # NCHW format
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = np.expand_dims(img_array, axis=0)

        return img_array.astype(np.float32)

    def dynamic_quantization(self):
        """Apply dynamic quantization to ResNet50"""
        print("Applying dynamic quantization...")

        quantize_dynamic(
            model_input=self.model_path,
            model_output="resnet50_dynamic_quant.onnx",
            weight_type=QuantType.QUInt8,
            per_channel=True,
            reduce_range=True,
            optimize_model=True
        )

        # Benchmark performance
        self.compare_models(self.model_path, "resnet50_dynamic_quant.onnx")

    def static_quantization_with_calibration(self):
        """Apply static quantization with calibration data"""
        print("Applying static quantization with calibration...")

        # Create calibration data reader
        calibration_reader = self.ResNetCalibrationDataReader(
            self.calibration_data_path, 100)

        quantize_static(
            model_input=self.model_path,
            model_output="resnet50_static_quant.onnx",
            calibration_data_reader=calibration_reader,
            quant_format=QuantFormat.QDQ,
            per_channel=True,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QUInt8,
            optimize_model=True
        )

        # Benchmark performance
        self.compare_models(self.model_path, "resnet50_static_quant.onnx")

    class ResNetCalibrationDataReader(CalibrationDataReader):
        def __init__(self, calibration_image_folder, num_samples=100):
            self.image_folder = calibration_image_folder
            self.num_samples = num_samples
            self.enum_data_dicts = []
            self._load_calibration_data()

        def _load_calibration_data(self):
            image_files = [f for f in os.listdir(self.image_folder)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            # Take only the required number of samples
            image_files = image_files[:self.num_samples]

            pipeline = ResNet50QuantizationPipeline()

            for image_file in image_files:
                image_path = os.path.join(self.image_folder, image_file)
                input_data = pipeline.preprocess_image(image_path)
                self.enum_data_dicts.append({"input": input_data})

        def get_next(self):
            if self.enum_data_dicts:
                return self.enum_data_dicts.pop(0)
            return None

    def compare_models(self, original_path, quantized_path):
        """Compare original and quantized models"""
        import time

        # Create sessions
        original_session = ort.InferenceSession(original_path)
        quantized_session = ort.InferenceSession(quantized_path)

        # Test data
        test_input = np.random.randn(1, 3, 224, 224).astype(np.float32)

        # Warmup
        for _ in range(10):
            original_session.run(None, {"input": test_input})
            quantized_session.run(None, {"input": test_input})

        # Benchmark
        num_runs = 100

        # Original model
        start_time = time.time()
        for _ in range(num_runs):
            orig_output = original_session.run(None, {"input": test_input})
        orig_time = (time.time() - start_time) / num_runs

        # Quantized model
        start_time = time.time()
        for _ in range(num_runs):
            quant_output = quantized_session.run(None, {"input": test_input})
        quant_time = (time.time() - start_time) / num_runs

        # Calculate accuracy difference
        accuracy_diff = np.mean(np.abs(orig_output[0] - quant_output[0]))

        # Get model sizes
        orig_size = os.path.getsize(original_path) / 1024 / 1024  # MB
        quant_size = os.path.getsize(quantized_path) / 1024 / 1024  # MB

        print(f"Performance Comparison:")
        print(f"Original model - Time: {orig_time*1000:.2f}ms, Size: {orig_size:.2f}MB")
        print(f"Quantized model - Time: {quant_time*1000:.2f}ms, Size: {quant_size:.2f}MB")
        print(f"Speedup: {orig_time/quant_time:.2f}x")
        print(f"Size reduction: {orig_size/quant_size:.2f}x")
        print(f"Average output difference: {accuracy_diff:.6f}")

# Usage example
def quantize_resnet50_example():
    pipeline = ResNet50QuantizationPipeline("resnet50.onnx")

    # Apply different quantization strategies
    pipeline.dynamic_quantization()
    pipeline.static_quantization_with_calibration()
```

#### Example 2: BERT Model Quantization
```python
class BERTQuantizationPipeline:
    def __init__(self, model_path="bert-base-uncased.onnx"):
        self.model_path = model_path
        self.tokenizer = self._load_tokenizer()

    def _load_tokenizer(self):
        """Load BERT tokenizer"""
        try:
            from transformers import BertTokenizer
            return BertTokenizer.from_pretrained('bert-base-uncased')
        except ImportError:
            print("Please install transformers: pip install transformers")
            return None

    def prepare_calibration_data(self, texts, max_length=128):
        """Prepare calibration data from text samples"""
        calibration_data = []

        for text in texts:
            # Tokenize text
            encoded = self.tokenizer.encode_plus(
                text,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='np'
            )

            calibration_data.append({
                'input_ids': encoded['input_ids'].astype(np.int64),
                'attention_mask': encoded['attention_mask'].astype(np.int64),
                'token_type_ids': encoded['token_type_ids'].astype(np.int64)
            })

        return calibration_data

    class BERTCalibrationDataReader(CalibrationDataReader):
        def __init__(self, calibration_data):
            self.calibration_data = calibration_data
            self.data_index = 0

        def get_next(self):
            if self.data_index < len(self.calibration_data):
                data = self.calibration_data[self.data_index]
                self.data_index += 1
                return data
            return None

    def selective_quantization(self):
        """Apply selective quantization to preserve accuracy"""
        # Sensitive layers that should not be quantized
        sensitive_nodes = [
            'bert/embeddings/LayerNorm/Add_1',
            'bert/encoder/layer.0/output/LayerNorm/Add_1',
            'bert/encoder/layer.11/output/LayerNorm/Add_1',
            'bert/pooler/dense/MatMul'
        ]

        # Get all quantizable nodes
        model = onnx.load(self.model_path)
        all_nodes = [node.name for node in model.graph.node
                    if node.op_type in ['MatMul', 'Gemm', 'Conv']]

        # Nodes to quantize (exclude sensitive ones)
        nodes_to_quantize = [node for node in all_nodes
                           if node not in sensitive_nodes]

        # Apply selective quantization
        quantize_dynamic(
            model_input=self.model_path,
            model_output="bert_selective_quant.onnx",
            weight_type=QuantType.QUInt8,
            nodes_to_quantize=nodes_to_quantize,
            per_channel=True,
            reduce_range=True
        )

        print(f"Quantized {len(nodes_to_quantize)} out of {len(all_nodes)} nodes")
        print(f"Preserved {len(sensitive_nodes)} sensitive nodes in FP32")

    def evaluate_quantization_impact(self, test_texts):
        """Evaluate the impact of quantization on model outputs"""
        original_session = ort.InferenceSession(self.model_path)
        quantized_session = ort.InferenceSession("bert_selective_quant.onnx")

        total_diff = 0
        max_diff = 0

        for text in test_texts:
            # Prepare input
            encoded = self.tokenizer.encode_plus(
                text, max_length=128, padding='max_length',
                truncation=True, return_tensors='np'
            )

            inputs = {
                'input_ids': encoded['input_ids'].astype(np.int64),
                'attention_mask': encoded['attention_mask'].astype(np.int64),
                'token_type_ids': encoded['token_type_ids'].astype(np.int64)
            }

            # Run inference
            orig_output = original_session.run(None, inputs)
            quant_output = quantized_session.run(None, inputs)

            # Calculate difference
            diff = np.mean(np.abs(orig_output[0] - quant_output[0]))
            total_diff += diff
            max_diff = max(max_diff, diff)

        avg_diff = total_diff / len(test_texts)
        print(f"Average output difference: {avg_diff:.6f}")
        print(f"Maximum output difference: {max_diff:.6f}")

        return avg_diff, max_diff

# Advanced quantization techniques
def advanced_quantization_techniques():
    """Demonstrate advanced quantization techniques"""

    # 1. Mixed-bit quantization
    def mixed_bit_quantization(model_path):
        """Use different bit widths for different layers"""
        # Load model to analyze layer sensitivity
        model = onnx.load(model_path)

        # Identify compute-intensive layers (can use lower precision)
        compute_layers = []
        sensitive_layers = []

        for node in model.graph.node:
            if node.op_type in ['MatMul', 'Gemm']:
                # Large matrix multiplications can use INT8
                compute_layers.append(node.name)
            elif node.op_type in ['LayerNormalization', 'Softmax']:
                # Normalization layers should stay in higher precision
                sensitive_layers.append(node.name)

        # First pass: 8-bit quantization for compute layers
        quantize_dynamic(
            model_input=model_path,
            model_output="temp_8bit.onnx",
            weight_type=QuantType.QUInt8,
            nodes_to_quantize=compute_layers
        )

        # Load the partially quantized model
        # Note: This is a simplified example - real implementation
        # would require more sophisticated handling
        print(f"Applied 8-bit quantization to {len(compute_layers)} compute layers")
        print(f"Keeping {len(sensitive_layers)} sensitive layers in FP32")

    # 2. Post-training quantization with knowledge distillation
    class QuantizationWithDistillation:
        def __init__(self, teacher_model_path, student_model_path):
            self.teacher_session = ort.InferenceSession(teacher_model_path)
            self.student_session = ort.InferenceSession(student_model_path)

        def distillation_loss(self, teacher_outputs, student_outputs, temperature=3.0):
            """Calculate knowledge distillation loss"""
            # Soften the outputs with temperature
            soft_teacher = np.exp(teacher_outputs / temperature)
            soft_teacher = soft_teacher / np.sum(soft_teacher, axis=-1, keepdims=True)

            soft_student = np.exp(student_outputs / temperature)
            soft_student = soft_student / np.sum(soft_student, axis=-1, keepdims=True)

            # KL divergence loss
            kl_loss = np.sum(soft_teacher * np.log(soft_teacher / soft_student))
            return kl_loss

        def evaluate_distillation(self, test_data):
            """Evaluate knowledge distillation quality"""
            total_loss = 0

            for batch in test_data:
                teacher_output = self.teacher_session.run(None, batch)
                student_output = self.student_session.run(None, batch)

                loss = self.distillation_loss(teacher_output[0], student_output[0])
                total_loss += loss

            avg_loss = total_loss / len(test_data)
            print(f"Average distillation loss: {avg_loss:.6f}")
            return avg_loss

    # 3. Adaptive quantization based on layer importance
    def adaptive_quantization(model_path, importance_scores):
        """Quantize layers based on importance scores"""
        model = onnx.load(model_path)

        # Sort layers by importance (lower score = less important = more quantizable)
        sorted_layers = sorted(importance_scores.items(), key=lambda x: x[1])

        # Quantize least important 80% of layers
        quantile_80 = int(0.8 * len(sorted_layers))
        layers_to_quantize = [layer for layer, _ in sorted_layers[:quantile_80]]

        quantize_dynamic(
            model_input=model_path,
            model_output="adaptive_quantized.onnx",
            weight_type=QuantType.QUInt8,
            nodes_to_quantize=layers_to_quantize,
            per_channel=True
        )

        print(f"Quantized {len(layers_to_quantize)} least important layers")
        return "adaptive_quantized.onnx"

# Example usage
def run_quantization_examples():
    # ResNet50 quantization
    resnet_pipeline = ResNet50QuantizationPipeline()
    resnet_pipeline.dynamic_quantization()

    # BERT quantization
    bert_pipeline = BERTQuantizationPipeline()
    bert_pipeline.selective_quantization()

    # Advanced techniques
    advanced_quantization_techniques()
```#### 2. Model Pruning:
```python
# Using ONNX simplifier
import onnxsim

model_simplified, check = onnxsim.simplify("model.onnx")
```

#### 3. Operator Fusion:
```cpp
// Custom fusion patterns
class CustomFusionPattern : public GraphTransformer {
    Status ApplyImpl(Graph& graph, bool& modified, int graph_level) const override {
        // Implement custom fusion logic
        return Status::OK();
    }
};
```

### Provider-Specific Optimizations

#### CUDA Optimizations:
```python
cuda_provider_options = {
    'arena_extend_strategy': 'kSameAsRequested',
    'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB limit
    'cudnn_conv_algo_search': 'HEURISTIC',     # Faster startup
    'do_copy_in_default_stream': True,         # Async operations
}
```

#### CPU Optimizations:
```python
# Set thread count
options = ort.SessionOptions()
options.intra_op_num_threads = 4  # Threads per operation
options.inter_op_num_threads = 2  # Parallel operations

# Enable OpenMP
options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
```

### Memory Optimization

#### Memory Pattern Optimization:
```python
# Enable memory pattern optimization
options = ort.SessionOptions()
options.enable_mem_pattern = True
options.enable_mem_reuse = True
options.add_session_config_entry("session.use_env_allocators", "1")
```

#### Custom Allocators:
```cpp
// Custom allocator implementation
class CustomAllocator : public IAllocator {
public:
    void* Alloc(size_t size) override {
        // Custom allocation logic
        return custom_malloc(size);
    }

    void Free(void* p) override {
        custom_free(p);
    }
};
```

### Batching Strategies

#### Dynamic Batching:
```python
# Model with dynamic batch size
session = ort.InferenceSession("dynamic_batch_model.onnx")

# Process multiple batch sizes efficiently
for batch_size in [1, 4, 8, 16]:
    input_data = np.random.randn(batch_size, 3, 224, 224)
    outputs = session.run(None, {"input": input_data})
```

#### Batch Processing:
```python
def process_batch(session, batch_data):
    """Efficient batch processing"""
    batch_size = len(batch_data)

    # Stack inputs
    batched_input = np.stack(batch_data)

    # Single inference call
    outputs = session.run(None, {"input": batched_input})

    # Split outputs
    return [outputs[0][i] for i in range(batch_size)]
```

### Benchmarking Framework

```python
class ONNXRuntimeBenchmark:
    def __init__(self, model_path, providers=None):
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def benchmark(self, input_data, num_runs=100, warmup_runs=10):
        # Warmup
        for _ in range(warmup_runs):
            self.session.run(None, {self.input_name: input_data})

        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.time()
            self.session.run(None, {self.input_name: input_data})
            times.append(time.time() - start)

        return {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times)
        }
```

---

## Chapter 9: Contributing to the Project

### Getting Started with Contributions

#### Prerequisites for Contributors:
1. **GitHub Account**: Set up with SSH keys
2. **Development Environment**: Follow Chapter 3 setup
3. **Code Editor**: VS Code recommended with C++ and Python extensions
4. **Testing Knowledge**: Familiarity with unit testing frameworks

### Development Workflow

#### 1. Fork and Clone:
```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/onnxruntime.git
cd onnxruntime
git remote add upstream https://github.com/microsoft/onnxruntime.git
```

#### 2. Create Feature Branch:
```bash
git checkout -b feature/your-feature-name
git push -u origin feature/your-feature-name
```

#### 3. Development Environment:
```bash
# Set up pre-commit hooks
pip install pre-commit
pre-commit install

# Build in debug mode for development
./build.sh --config Debug --enable_logging
```

### Code Standards and Guidelines

#### C++ Coding Standards:
- Follow Google C++ Style Guide
- Use clang-format for formatting
- Include comprehensive unit tests
- Document public APIs

#### Python Standards:
- Follow PEP 8 style guide
- Use black for code formatting
- Type hints for public functions
- Comprehensive docstrings

#### Example C++ Contribution:
```cpp
// include/onnxruntime/core/providers/cpu/math/gemm.h
#pragma once
#include "core/framework/op_kernel.h"

namespace onnxruntime {

class Gemm final : public OpKernel {
 public:
  explicit Gemm(const OpKernelInfo& info);
  Status Compute(OpKernelContext* context) const override;

 private:
  float alpha_;
  float beta_;
  int64_t trans_a_;
  int64_t trans_b_;
};

} // namespace onnxruntime
```

### Testing Requirements

#### Unit Tests:
```cpp
// onnxruntime/test/providers/cpu/math/gemm_test.cc
#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

TEST(GemmTest, BasicTest) {
  OpTester test("Gemm");
  test.AddAttribute("alpha", 1.0f);
  test.AddAttribute("beta", 1.0f);

  test.AddInput<float>("A", {2, 3}, {1, 2, 3, 4, 5, 6});
  test.AddInput<float>("B", {3, 2}, {1, 2, 3, 4, 5, 6});
  test.AddInput<float>("C", {2, 2}, {1, 1, 1, 1});

  test.AddOutput<float>("Y", {2, 2}, {23, 31, 53, 71});
  test.Run();
}

} // namespace test
} // namespace onnxruntime
```

#### Python Tests:
```python
# onnxruntime/test/python/test_backend.py
import unittest
import numpy as np
import onnxruntime as ort

class TestBackend(unittest.TestCase):
    def test_basic_inference(self):
        session = ort.InferenceSession("test_model.onnx")
        input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)

        outputs = session.run(None, {"input": input_data})

        self.assertIsNotNone(outputs)
        self.assertEqual(len(outputs), 1)
```

### Types of Contributions

#### 1. Bug Fixes:
- Identify issue through GitHub issues
- Create minimal reproduction case
- Implement fix with tests
- Ensure no performance regression

#### 2. New Operator Support:
```cpp
// Adding a new operator
class CustomOp final : public OpKernel {
 public:
  explicit CustomOp(const OpKernelInfo& info) : OpKernel(info) {
    // Initialize operator parameters
  }

  Status Compute(OpKernelContext* context) const override {
    // Implement operator logic
    return Status::OK();
  }
};

// Register operator
ONNX_OPERATOR_KERNEL_EX(
    CustomOp,
    kOnnxDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder(),
    CustomOp);
```

#### 3. Execution Provider Support:
```cpp
// Custom execution provider
class CustomExecutionProvider : public IExecutionProvider {
 public:
  CustomExecutionProvider(const CustomExecutionProviderInfo& info)
      : IExecutionProvider{kCustomExecutionProvider} {
    // Initialize provider
  }

  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const GraphViewer& graph_viewer,
                const IKernelLookup& kernel_lookup) const override {
    // Return supported operations
  }
};
```

#### 4. Performance Improvements:
- Profile existing code
- Identify bottlenecks
- Implement optimizations
- Benchmark improvements
- Document performance gains

### Pull Request Process

#### 1. PR Guidelines:
- **Title**: Clear, descriptive summary
- **Description**: Detailed explanation of changes
- **Testing**: Include test results and benchmarks
- **Documentation**: Update relevant docs
- **Breaking Changes**: Clearly document any breaking changes

#### 2. PR Template:
```markdown
## Description
Brief description of changes

## Motivation and Context
Why is this change required? What problem does it solve?

## How Has This Been Tested?
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Performance benchmarks

## Types of changes
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to change)

## Checklist:
- [ ] My code follows the code style of this project
- [ ] My change requires a change to the documentation
- [ ] I have updated the documentation accordingly
- [ ] I have added tests to cover my changes
```

#### 3. Review Process:
1. **Automated Checks**: CI/CD pipeline runs tests
2. **Code Review**: Team members review code
3. **Performance Review**: Check for performance impacts
4. **Documentation Review**: Ensure docs are updated
5. **Final Approval**: Maintainer approval required

### Community Guidelines

#### Communication Channels:
- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and discussions
- **Discord**: Real-time community chat
- **Stack Overflow**: Technical questions with `onnxruntime` tag

#### Code of Conduct:
- Be respectful and inclusive
- Provide constructive feedback
- Help newcomers to the project
- Follow Microsoft Open Source Code of Conduct

---

## Chapter 10: Advanced Topics

### Custom Operators

Creating custom operators allows you to extend ONNX Runtime with domain-specific operations.

#### C++ Custom Operator:
```cpp
// custom_op.cc
#include "onnxruntime_cxx_api.h"

struct CustomOpKernel {
    CustomOpKernel(const OrtKernelInfo* info) {
        // Initialize kernel parameters
    }

    void Compute(OrtKernelContext* context) {
        // Get input tensors
        const OrtValue* input = ort_api->KernelContext_GetInput(context, 0);
        const float* input_data = nullptr;
        ort_api->GetTensorMutableData(input, (void**)&input_data);

        // Get output tensor
        OrtValue* output = nullptr;
        int64_t output_shape[] = {1, 10};
        ort_api->KernelContext_GetOutput(context, 0, output_shape, 2, &output);
        float* output_data = nullptr;
        ort_api->GetTensorMutableData(output, (void**)&output_data);

        // Implement custom operation
        for (int i = 0; i < 10; ++i) {
            output_data[i] = input_data[i] * 2.0f;  // Example operation
        }
    }
};

// Operator definition
static const char* c_OpDomain = "custom.domain";
static const char* c_CustomOp = "CustomOp";

static OrtCustomOp c_CustomOpInstance = {
    ORT_API_VERSION,
    c_CustomOp,
    c_OpDomain,
    1,  // version
    nullptr,  // GetInputTypeCount
    nullptr,  // GetInputType
    nullptr,  // GetOutputTypeCount
    nullptr,  // GetOutputType
    CustomOpKernel::Create,
    CustomOpKernel::Destroy,
    CustomOpKernel::Compute,
    nullptr,  // GetInputCharacteristic
    nullptr   // GetOutputCharacteristic
};
```

#### Python Custom Operator:
```python
import numpy as np
from onnxruntime.capi._pybind_state import CustomOpDef

class CustomPyOp:
    def __init__(self):
        pass

    def compute(self, input_data):
        # Custom computation
        return input_data * 2.0

# Register custom operator
def register_custom_ops():
    custom_op = CustomOpDef(
        op_type="CustomPyOp",
        domain="custom.domain",
        inputs=[CustomOpDef.TensorType.FLOAT],
        outputs=[CustomOpDef.TensorType.FLOAT]
    )
    return [custom_op]
```

### Graph Transformers

Graph transformers optimize the computation graph for better performance.

#### Custom Graph Transformer:
```cpp
// custom_transformer.h
#include "core/optimizer/graph_transformer.h"

class CustomTransformer : public GraphTransformer {
 public:
  CustomTransformer() : GraphTransformer("CustomTransformer") {}

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level) const override {
    auto& nodes = graph.Nodes();

    for (auto& node : nodes) {
      if (CanOptimizeNode(node)) {
        OptimizeNode(graph, node);
        modified = true;
      }
    }

    return Status::OK();
  }

 private:
  bool CanOptimizeNode(const Node& node) const {
    // Check if node can be optimized
    return node.OpType() == "Conv" && HasSpecificPattern(node);
  }

  void OptimizeNode(Graph& graph, Node& node) const {
    // Implement optimization logic
  }
};
```

### Memory Planning

Advanced memory management for optimal performance:

#### Custom Memory Planner:
```cpp
class CustomMemoryPlanner : public IMemoryPlanner {
 public:
  Status CreatePlan(
      const std::vector<const NodeArg*>& inputs,
      const std::vector<const NodeArg*>& outputs,
      const std::unordered_map<std::string, TensorShape>& outer_scope_node_arg_to_shape_map,
      const ExecutionPlanBase& execution_plan,
      gsl::span<const int> execution_plan_to_node_index,
      const logging::Logger& logger) override {

    // Implement custom memory planning logic
    return Status::OK();
  }
};
```

### Quantization and Pruning

#### Dynamic Quantization:
```python
from onnxruntime.quantization import quantize_dynamic, QuantType

# Quantize model
quantized_model = quantize_dynamic(
    model_input="model.onnx",
    model_output="model_quantized.onnx",
    weight_type=QuantType.QUInt8,
    per_channel=True,
    reduce_range=True
)
```

#### Static Quantization:
```python
from onnxruntime.quantization import quantize_static, CalibrationDataReader

class DataReader(CalibrationDataReader):
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.preprocess_function = preprocess_func
        self.enum_data_dicts = iter([{"input": data} for data in calibration_data])

    def get_next(self):
        return next(self.enum_data_dicts, None)

# Static quantization with calibration data
quantize_static(
    model_input="model.onnx",
    model_output="model_quantized.onnx",
    calibration_data_reader=DataReader("calibration_data/"),
    quant_format=QuantFormat.QDQ
)
```

### Multi-Threading and Parallelization

#### Custom Thread Pool:
```cpp
class CustomThreadPool {
 public:
  CustomThreadPool(int num_threads) : num_threads_(num_threads) {
    for (int i = 0; i < num_threads_; ++i) {
      workers_.emplace_back([this] {
        while (!stop_) {
          std::function<void()> task;
          {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            condition_.wait(lock, [this] { return stop_ || !tasks_.empty(); });

            if (stop_ && tasks_.empty()) return;

            task = std::move(tasks_.front());
            tasks_.pop();
          }
          task();
        }
      });
    }
  }

  template<class F>
  void Enqueue(F&& f) {
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      tasks_.emplace(std::forward<F>(f));
    }
    condition_.notify_one();
  }

 private:
  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> tasks_;
  std::mutex queue_mutex_;
  std::condition_variable condition_;
  bool stop_ = false;
  int num_threads_;
};
```

### WebAssembly Support

#### Building for WebAssembly:
```bash
# Install Emscripten
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk
./emsdk install latest
./emsdk activate latest

# Build ONNX Runtime for WebAssembly
python tools/ci_build/build.py \
    --build_wasm \
    --config Release \
    --enable_wasm_simd \
    --disable_rtti \
    --minimal_build extended
```

#### Using in Web Applications:
```javascript
// Load ONNX Runtime WebAssembly
import * as ort from 'onnxruntime-web/webassembly';

// Configure WebAssembly
ort.env.wasm.wasmPaths = '/path/to/wasm/files/';
ort.env.wasm.numThreads = 4;
ort.env.wasm.simd = true;

// Create session and run inference
const session = await ort.InferenceSession.create('model.onnx');
const feeds = { input: new ort.Tensor('float32', inputData, inputShape) };
const results = await session.run(feeds);
```

#### Advanced Quantization Techniques: INT4, BF16, and Training Integration

#### 1. INT4 Quantization (4-bit)

```python
class INT4Quantization:
    """Ultra-low precision 4-bit quantization for extreme compression"""

    def __init__(self):
        self.bit_width = 4
        self.quant_min = -8  # -2^(4-1)
        self.quant_max = 7   # 2^(4-1) - 1

    def apply_int4_quantization(self, model_path, output_path):
        """Apply 4-bit quantization with custom calibration"""

        # Custom INT4 quantization since ONNX Runtime doesn't natively support INT4
        model = onnx.load(model_path)

        # Modify each quantizable node
        for node in model.graph.node:
            if self._is_quantizable_node(node):
                self._quantize_node_to_int4(node, model.graph)

        # Save INT4 quantized model
        onnx.save(model, output_path)

        return self._validate_int4_model(model_path, output_path)

    def _quantize_weights_int4(self, weights):
        """Quantize weights to 4-bit integers"""
        # Calculate scale and zero point for INT4 range
        w_min, w_max = np.min(weights), np.max(weights)
        scale = (w_max - w_min) / (self.quant_max - self.quant_min)
        zero_point = self.quant_min - w_min / scale
        zero_point = np.round(np.clip(zero_point, self.quant_min, self.quant_max))

        # Quantize
        quantized_weights = np.round((weights / scale) + zero_point)
        quantized_weights = np.clip(quantized_weights, self.quant_min, self.quant_max)

        # Dequantize for storage (since ONNX Runtime will handle as FP32)
        dequantized_weights = scale * (quantized_weights - zero_point)

        return dequantized_weights, scale, zero_point

    def group_wise_int4_quantization(self, model_path, group_size=128):
        """Group-wise INT4 quantization for better accuracy"""
        model = onnx.load(model_path)

        for node in model.graph.node:
            if node.op_type in ['Conv', 'MatMul']:
                weights = self._extract_weights(node, model.graph)

                if weights.size > group_size:
                    # Quantize in groups
                    quantized_groups = []
                    scales = []
                    zero_points = []

                    for i in range(0, weights.size, group_size):
                        group = weights.flat[i:i+group_size]
                        q_group, scale, zp = self._quantize_weights_int4(group.reshape(-1))

                        quantized_groups.append(q_group)
                        scales.append(scale)
                        zero_points.append(zp)

                    # Reconstruct weights
                    reconstructed_weights = np.concatenate(quantized_groups).reshape(weights.shape)
                    self._update_node_weights(node, reconstructed_weights, model.graph)

        onnx.save(model, f"{model_path.replace('.onnx', '_int4_group.onnx')}")
        return scales, zero_points

#### Example: BERT INT4 Quantization
class BERTINT4Quantizer:
    def __init__(self):
        self.sensitive_layers = [
            'attention.output.dense',
            'output.LayerNorm',
            'attention.output.LayerNorm'
        ]

    def quantize_bert_int4(self, model_path):
        """Selective INT4 quantization for BERT"""
        int4_quantizer = INT4Quantization()

        # Load BERT model
        model = onnx.load(model_path)

        # Apply INT4 to non-sensitive layers only
        for node in model.graph.node:
            layer_name = self._get_layer_name(node)

            if (node.op_type in ['MatMul', 'Gemm'] and
                layer_name not in self.sensitive_layers):

                # Apply group-wise INT4 quantization
                weights = self._extract_weights(node, model.graph)
                q_weights, scale, zp = int4_quantizer._quantize_weights_int4(weights)
                self._update_node_weights(node, q_weights, model.graph)

                print(f"Quantized {layer_name} to INT4")

        # Save selective INT4 model
        output_path = model_path.replace('.onnx', '_selective_int4.onnx')
        onnx.save(model, output_path)

        return output_path
```

#### 2. BF16 (Brain Floating Point) Implementation

```python
class BF16Quantization:
    """BF16 quantization with same dynamic range as FP32 but reduced precision"""

    def __init__(self):
        self.mantissa_bits = 7  # vs 23 in FP32
        self.exponent_bits = 8  # same as FP32
        self.total_bits = 16

    def convert_to_bf16(self, model_path, output_path, preserve_ops=None):
        """Convert model to BF16 with selective preservation"""

        if preserve_ops is None:
            preserve_ops = ['Softmax', 'LayerNormalization', 'Sigmoid', 'Tanh']

        model = onnx.load(model_path)

        # Convert tensors to BF16
        for tensor in model.graph.initializer:
            if tensor.data_type == onnx.TensorProto.FLOAT:
                # Convert FP32 to BF16
                fp32_data = onnx.numpy_helper.to_array(tensor)
                bf16_data = self._fp32_to_bf16(fp32_data)

                # Update tensor
                tensor.ClearField('float_data')
                tensor.ClearField('raw_data')
                tensor.raw_data = bf16_data.tobytes()
                tensor.data_type = onnx.TensorProto.BFLOAT16

        # Update nodes that should remain in FP32
        for node in model.graph.node:
            if node.op_type in preserve_ops:
                self._preserve_node_precision(node, model.graph)

        onnx.save(model, output_path)

        return self._benchmark_bf16_model(model_path, output_path)

    def _fp32_to_bf16(self, fp32_array):
        """Convert FP32 array to BF16"""
        # BF16 truncates FP32 mantissa from 23 to 7 bits
        # Simple implementation: round to nearest even

        # View as uint32
        uint32_view = fp32_array.view(np.uint32)

        # BF16 conversion: keep sign + exponent + top 7 mantissa bits
        # Round to nearest even (add 0x7FFF then truncate)
        rounding_bias = 0x7FFF + ((uint32_view >> 16) & 1)
        bf16_bits = (uint32_view + rounding_bias) >> 16

        return bf16_bits.astype(np.uint16)

    def mixed_bf16_fp32_training(self, model_path, loss_scale=2**15):
        """Mixed BF16/FP32 training implementation"""

        class BF16TrainingSession:
            def __init__(self, model_path):
                self.model_path = model_path
                self.loss_scale = loss_scale
                self.bf16_ops = ['Conv', 'MatMul', 'Gemm', 'Add', 'Mul']
                self.fp32_ops = ['Softmax', 'LayerNormalization', 'CrossEntropy']

            def forward_pass_bf16(self, inputs):
                """Forward pass with mixed BF16/FP32"""
                # Convert inputs to BF16 for compatible operations
                bf16_inputs = self._convert_inputs_bf16(inputs)

                # Run inference with mixed precision
                session = onnxruntime.InferenceSession(self.model_path)
                outputs = session.run(None, bf16_inputs)

                return outputs

            def backward_pass_with_scaling(self, loss, gradients):
                """Backward pass with gradient scaling for BF16"""
                # Scale loss to prevent gradient underflow
                scaled_loss = loss * self.loss_scale

                # Compute gradients
                scaled_gradients = self._compute_gradients(scaled_loss)

                # Unscale gradients
                unscaled_gradients = {
                    name: grad / self.loss_scale
                    for name, grad in scaled_gradients.items()
                }

                # Check for gradient overflow/underflow
                if self._check_gradient_overflow(unscaled_gradients):
                    self.loss_scale /= 2  # Reduce scale
                    return None  # Skip this update

                return unscaled_gradients

        return BF16TrainingSession(model_path)

#### Example: Vision Transformer BF16 Training
class ViTBF16Training:
    def __init__(self):
        self.bf16_quantizer = BF16Quantization()

    def train_vit_bf16(self, model_path, train_loader, epochs=10):
        """Train Vision Transformer with BF16 mixed precision"""

        # Convert model to BF16
        bf16_model_path = model_path.replace('.onnx', '_bf16.onnx')
        self.bf16_quantizer.convert_to_bf16(
            model_path, bf16_model_path,
            preserve_ops=['LayerNormalization', 'Softmax', 'GELU']
        )

        # Training session with BF16
        training_session = self.bf16_quantizer.mixed_bf16_fp32_training(bf16_model_path)

        for epoch in range(epochs):
            epoch_loss = 0

            for batch_idx, (data, targets) in enumerate(train_loader):
                # Forward pass in BF16
                outputs = training_session.forward_pass_bf16({'input': data})

                # Compute loss (in FP32 for stability)
                loss = self._compute_loss(outputs, targets)

                # Backward pass with gradient scaling
                gradients = training_session.backward_pass_with_scaling(loss, outputs)

                if gradients is not None:  # No overflow
                    self._update_weights(gradients)

                epoch_loss += loss.item()

            print(f"Epoch {epoch}: Loss = {epoch_loss / len(train_loader):.4f}")

        return bf16_model_path
```

#### 3. Quantization-Aware Training (QAT) Integration

```python
class QuantizationAwareTrainingPipeline:
    """Complete QAT pipeline for training with quantization"""

    def __init__(self, precision_config):
        self.precision_config = precision_config  # {'conv': 'int8', 'attention': 'int16'}
        self.quantization_schedulers = {}

    def setup_qat_training(self, model_path, train_config):
        """Setup complete QAT training pipeline"""

        # 1. Prepare model for QAT
        qat_model = self._prepare_model_for_qat(model_path)

        # 2. Setup quantization observers
        self._setup_quantization_observers(qat_model)

        # 3. Configure training loop
        training_session = self._configure_training_session(qat_model, train_config)

        return training_session

    def progressive_quantization_training(self, model_path, train_loader, val_loader):
        """Progressive quantization during training"""

        # Phase 1: Full precision warmup (epochs 0-2)
        print("Phase 1: Full precision warmup...")
        model = self._train_full_precision(model_path, train_loader, epochs=3)

        # Phase 2: FP16 mixed precision (epochs 3-5)
        print("Phase 2: FP16 mixed precision...")
        fp16_model = self._enable_fp16_training(model)
        fp16_model = self._train_with_precision(fp16_model, train_loader, epochs=3, precision='fp16')

        # Phase 3: INT16 quantization for sensitive layers (epochs 6-8)
        print("Phase 3: INT16 for sensitive layers...")
        int16_model = self._apply_selective_int16(fp16_model, self.precision_config)
        int16_model = self._train_with_precision(int16_model, train_loader, epochs=3, precision='int16')

        # Phase 4: INT8 quantization for robust layers (epochs 9-12)
        print("Phase 4: INT8 for robust layers...")
        int8_model = self._apply_selective_int8(int16_model, self.precision_config)
        int8_model = self._train_with_precision(int8_model, train_loader, epochs=4, precision='int8')

        # Phase 5: Fine-tuning with quantization noise (epochs 13-15)
        print("Phase 5: Fine-tuning with quantization noise...")
        final_model = self._fine_tune_with_noise(int8_model, train_loader, epochs=3)

        return final_model

    def knowledge_distillation_qat(self, teacher_model_path, student_model_path,
                                  train_loader, temperature=4.0):
        """QAT with knowledge distillation from full-precision teacher"""

        # Load teacher (full precision) and student (quantized) models
        teacher_session = onnxruntime.InferenceSession(teacher_model_path)
        student_model = self._prepare_model_for_qat(student_model_path)

        # Training loop with distillation
        for epoch in range(15):
            student_model.train()

            for batch_idx, (data, targets) in enumerate(train_loader):
                # Teacher forward pass (full precision)
                with torch.no_grad():
                    teacher_outputs = teacher_session.run(None, {'input': data.numpy()})
                    teacher_logits = torch.from_numpy(teacher_outputs[0])
                    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)

                # Student forward pass (quantized)
                student_logits = student_model(data)
                student_log_probs = F.log_softmax(student_logits / temperature, dim=1)

                # Combined loss: distillation + task loss
                distillation_loss = F.kl_div(
                    student_log_probs, teacher_probs, reduction='batchmean'
                ) * (temperature ** 2)

                task_loss = F.cross_entropy(student_logits, targets)
                total_loss = 0.7 * distillation_loss + 0.3 * task_loss

                # Backward pass with quantization-aware gradients
                total_loss.backward()
                self._update_quantization_parameters(student_model)

        return student_model

    def adaptive_bit_width_training(self, model_path, train_loader, target_size_mb):
        """Train with adaptive bit-width selection"""

        # Start with mixed precision baseline
        current_model = self._prepare_mixed_precision_model(model_path)
        current_size = self._calculate_model_size(current_model)

        # Bit-width search during training
        layer_bit_widths = {layer: 16 for layer in self._get_quantizable_layers(current_model)}

        for epoch in range(20):
            # Training step
            epoch_loss = self._train_epoch(current_model, train_loader)

            # Evaluate current model
            accuracy = self._evaluate_model(current_model)

            # Adaptive bit-width adjustment every 3 epochs
            if epoch % 3 == 0 and epoch > 0:
                layer_bit_widths = self._adjust_bit_widths(
                    current_model, layer_bit_widths, target_size_mb, accuracy
                )

                # Apply new bit-widths
                current_model = self._apply_bit_width_config(current_model, layer_bit_widths)

            print(f"Epoch {epoch}: Loss={epoch_loss:.4f}, Accuracy={accuracy:.4f}, "
                  f"Size={self._calculate_model_size(current_model):.1f}MB")

        return current_model, layer_bit_widths

#### Example: Complete BERT QAT Training
class BERTQuantizationAwareTraining:
    def __init__(self):
        self.qat_pipeline = QuantizationAwareTrainingPipeline({
            'embeddings': 'int8',
            'encoder_layers': 'int8',
            'attention': 'int16',  # More sensitive
            'layer_norm': 'fp16',  # Very sensitive
            'classifier': 'int16'
        })

    def train_bert_qat(self, model_path, train_loader, val_loader):
        """Complete BERT QAT training example"""

        print("Starting BERT Quantization-Aware Training...")

        # Setup QAT training
        training_session = self.qat_pipeline.setup_qat_training(
            model_path,
            {'optimizer': 'adamw', 'lr': 2e-5, 'weight_decay': 0.01}
        )

        # Progressive quantization training
        bert_qat_model = self.qat_pipeline.progressive_quantization_training(
            model_path, train_loader, val_loader
        )

        # Convert to fully quantized model
        final_quantized_model = self._convert_qat_to_quantized(bert_qat_model)

        # Benchmark results
        results = self._benchmark_bert_quantization(
            original_model=model_path,
            quantized_model=final_quantized_model,
            test_data=val_loader
        )

        print(f"QAT Results:")
        print(f"  Model size reduction: {results['size_reduction']:.1f}x")
        print(f"  Inference speedup: {results['speedup']:.1f}x")
        print(f"  Accuracy preservation: {results['accuracy_retention']:.1%}")

        return final_quantized_model, results
```

#### 4. Production Deployment Quantization

```python
class ProductionQuantizationDeployment:
    """Production-ready quantization for different deployment scenarios"""

    def __init__(self):
        self.deployment_configs = {
            'server_cpu': {
                'target_latency': 10,  # ms
                'batch_size': 32,
                'precision': 'int8',
                'optimization_level': 'aggressive'
            },
            'server_gpu': {
                'target_latency': 5,   # ms
                'batch_size': 64,
                'precision': 'fp16',
                'optimization_level': 'all'
            },
            'mobile': {
                'target_latency': 100, # ms
                'model_size_limit': 50, # MB
                'precision': 'int8',
                'optimization_level': 'basic'
            },
            'edge': {
                'target_latency': 200, # ms
                'model_size_limit': 10, # MB
                'precision': 'int4',
                'optimization_level': 'extreme'
            }
        }

    def deploy_quantized_model(self, model_path, deployment_target, validation_data):
        """Deploy quantized model for specific target"""

        config = self.deployment_configs[deployment_target]

        # Apply target-specific quantization
        if config['precision'] == 'int8':
            quantized_model = self._apply_int8_quantization(model_path, config)
        elif config['precision'] == 'fp16':
            quantized_model = self._apply_fp16_quantization(model_path, config)
        elif config['precision'] == 'int4':
            quantized_model = self._apply_int4_quantization(model_path, config)
        elif config['precision'] == 'bf16':
            quantized_model = self._apply_bf16_quantization(model_path, config)

        # Validate deployment constraints
        validation_results = self._validate_deployment_constraints(
            quantized_model, config, validation_data)

        if validation_results['meets_constraints']:
            # Final optimization for deployment
            optimized_model = self._optimize_for_deployment(quantized_model, config)

            return {
                'success': True,
                'model_path': optimized_model,
                'performance_metrics': validation_results['metrics'],
                'deployment_config': config
            }
        else:
            # Suggest alternative configurations
            return {
                'success': False,
                'failed_constraints': validation_results['failed_constraints'],
                'suggestions': self._suggest_deployment_alternatives(config, validation_results)
            }

    def benchmark_across_hardware(self, model_path, quantization_types=['fp32', 'fp16', 'bf16', 'int8']):
        """Benchmark model across different hardware and quantization types"""

        hardware_targets = ['cpu_x86', 'cpu_arm', 'gpu_nvidia', 'gpu_amd', 'mobile_android', 'mobile_ios']
        results_matrix = {}

        for hardware in hardware_targets:
            results_matrix[hardware] = {}

            for quant_type in quantization_types:
                try:
                    # Apply quantization
                    quantized_path = f"temp_{hardware}_{quant_type}.onnx"
                    self._apply_quantization_for_hardware(model_path, quantized_path, quant_type, hardware)

                    # Benchmark on target hardware (simulated)
                    performance = self._benchmark_on_hardware(quantized_path, hardware)

                    results_matrix[hardware][quant_type] = {
                        'latency_ms': performance['latency'],
                        'throughput_qps': performance['throughput'],
                        'memory_mb': performance['memory'],
                        'accuracy': performance['accuracy'],
                        'model_size_mb': performance['model_size']
                    }

                except Exception as e:
                    results_matrix[hardware][quant_type] = {'error': str(e)}

        # Generate recommendations
        recommendations = self._generate_deployment_recommendations(results_matrix)

        return {
            'benchmark_results': results_matrix,
            'recommendations': recommendations,
            'best_configs': self._find_optimal_configs(results_matrix)
        }

# Example usage for production deployment
def deploy_bert_for_production():
    """Example: Deploy BERT model for production with quantization"""

    deployment_manager = ProductionQuantizationDeployment()

    # Benchmark across different deployment scenarios
    benchmark_results = deployment_manager.benchmark_across_hardware(
        "bert-base-uncased.onnx",
        quantization_types=['fp32', 'fp16', 'bf16', 'int8']
    )

    print("Benchmark Results:")
    for hardware, results in benchmark_results['benchmark_results'].items():
        print(f"\n{hardware}:")
        for quant_type, metrics in results.items():
            if 'error' not in metrics:
                print(f"  {quant_type}: {metrics['latency_ms']:.1f}ms, "
                      f"{metrics['throughput_qps']:.1f} QPS, "
                      f"{metrics['accuracy']:.3f} acc")

    print(f"\nRecommendations: {benchmark_results['recommendations']}")

    # Deploy for specific targets
    server_deployment = deployment_manager.deploy_quantized_model(
        "bert-base-uncased.onnx", "server_gpu", validation_data
    )

    mobile_deployment = deployment_manager.deploy_quantized_model(
        "bert-base-uncased.onnx", "mobile", validation_data
    )

    return server_deployment, mobile_deployment


### Mixed-Precision Training: Theory and Implementation

Mixed-precision training uses both 16-bit (FP16) and 32-bit (FP32) floating-point representations to accelerate training while maintaining model accuracy.

#### Theoretical Background

**IEEE 754 Floating Point Formats**:
- **FP32**: 1 sign + 8 exponent + 23 mantissa bits (range: ~10^-38 to 10^38)
- **FP16**: 1 sign + 5 exponent + 10 mantissa bits (range: ~10^-8 to 65504)
- **BF16**: 1 sign + 8 exponent + 7 mantissa bits (range: same as FP32, less precision)

#### Automatic Mixed Precision (AMP)

```python
import onnxruntime.training as orttraining
import torch
import numpy as np

class MixedPrecisionTrainer:
    def __init__(self, model_path, device='cuda'):
        self.device = device

        # Initialize ONNX Runtime training session with mixed precision
        self.training_session = orttraining.TrainingSession(
            train_model_path=model_path,
            eval_model_path=model_path.replace('training', 'eval'),
            optimizer_model_path=model_path.replace('training', 'optimizer'),
            device_id=0 if device == 'cuda' else -1
        )

        # Enable mixed precision
        self.training_session.set_seed(42)

        # Gradient scaler for FP16 training
        self.scaler = orttraining.amp.GradScaler()

    def train_step_with_amp(self, inputs, labels):
        """Training step with Automatic Mixed Precision"""

        # Convert inputs to appropriate precision
        fp16_inputs = {k: v.half() if v.dtype == torch.float32 else v
                      for k, v in inputs.items()}

        # Forward pass in mixed precision
        with orttraining.amp.autocast():
            loss = self.training_session.train_step(fp16_inputs, labels)

        # Scale loss to prevent gradient underflow
        scaled_loss = self.scaler.scale(loss)

        # Backward pass
        self.training_session.lazy_reset_grad()
        scaled_loss.backward()

        # Unscale gradients and update
        self.scaler.step(self.training_session.get_optimizer())
        self.scaler.update()

        return loss.item()

# Example usage
def mixed_precision_training_example():
    trainer = MixedPrecisionTrainer("training_model.onnx")

    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            # Prepare inputs
            inputs = {"input": data}
            labels = {"target": target}

            # Training step with AMP
            loss = trainer.train_step_with_amp(inputs, labels)

            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss:.6f}')
```

#### Manual Mixed Precision Implementation

```python
class ManualMixedPrecisionTrainer:
    def __init__(self, model_path):
        self.training_session = orttraining.TrainingSession(
            train_model_path=model_path,
            eval_model_path=model_path.replace('training', 'eval'),
            optimizer_model_path=model_path.replace('training', 'optimizer')
        )

        # Gradient scaling parameters
        self.loss_scale = 2**15  # Initial loss scale
        self.scale_factor = 2.0
        self.scale_window = 2000
        self.unskipped_steps = 0

    def dynamic_loss_scaling(self, gradients):
        """Implement dynamic loss scaling"""
        # Check for gradient overflow
        grad_norm = self._compute_gradient_norm(gradients)

        if torch.isfinite(grad_norm):
            # No overflow, proceed with update
            self.unskipped_steps += 1

            # Increase scale if no overflow for scale_window steps
            if self.unskipped_steps >= self.scale_window:
                self.loss_scale *= self.scale_factor
                self.unskipped_steps = 0

            return True, grad_norm
        else:
            # Overflow detected, reduce scale
            self.loss_scale /= self.scale_factor
            self.unskipped_steps = 0
            return False, float('inf')

    def train_step_manual_amp(self, inputs, labels):
        """Manual mixed precision training step"""

        # Convert inputs to FP16
        fp16_inputs = self._to_fp16(inputs)

        # Forward pass in FP16
        loss = self.training_session.train_step(fp16_inputs, labels)

        # Scale loss
        scaled_loss = loss * self.loss_scale

        # Backward pass
        self.training_session.lazy_reset_grad()
        scaled_loss.backward()

        # Get gradients
        gradients = self._get_gradients()

        # Unscale gradients
        unscaled_gradients = [g / self.loss_scale for g in gradients]

        # Dynamic loss scaling
        should_update, grad_norm = self.dynamic_loss_scaling(unscaled_gradients)

        if should_update:
            # Clip gradients if necessary
            if grad_norm > self.max_grad_norm:
                self._clip_gradients(unscaled_gradients, self.max_grad_norm)

            # Update weights
            self.training_session.optimizer_step()

        return loss.item(), should_update
```

#### Layer-wise Precision Control

```python
class LayerWisePrecisionController:
    def __init__(self):
        self.precision_policy = {
            # Keep embeddings and layer norms in FP32 for stability
            'embedding': torch.float32,
            'layer_norm': torch.float32,
            'batch_norm': torch.float32,

            # Use FP16 for compute-intensive layers
            'linear': torch.float16,
            'conv2d': torch.float16,
            'attention': torch.float16,

            # Keep final classification layer in FP32
            'classifier': torch.float32
        }

    def apply_precision_policy(self, model):
        """Apply layer-wise precision policy"""
        for name, module in model.named_modules():
            layer_type = self._get_layer_type(module)
            target_dtype = self.precision_policy.get(layer_type, torch.float16)

            if target_dtype == torch.float16:
                module.half()
            else:
                module.float()

    def _get_layer_type(self, module):
        """Determine layer type for precision policy"""
        class_name = module.__class__.__name__.lower()

        if 'embedding' in class_name:
            return 'embedding'
        elif 'norm' in class_name:
            return 'layer_norm'
        elif 'linear' in class_name or 'dense' in class_name:
            return 'linear'
        elif 'conv' in class_name:
            return 'conv2d'
        elif 'attention' in class_name:
            return 'attention'
        elif 'classifier' in class_name:
            return 'classifier'
        else:
            return 'default'
```

#### Advanced Mixed Precision Techniques

#### 1. Gradient Accumulation with Mixed Precision

```python
class GradientAccumulationAMP:
    def __init__(self, model_path, accumulation_steps=4):
        self.training_session = orttraining.TrainingSession(
            train_model_path=model_path,
            eval_model_path=model_path.replace('training', 'eval'),
            optimizer_model_path=model_path.replace('training', 'optimizer')
        )
        self.accumulation_steps = accumulation_steps
        self.scaler = orttraining.amp.GradScaler()

    def train_with_accumulation(self, data_loader):
        """Training with gradient accumulation and mixed precision"""

        for batch_idx, (inputs, labels) in enumerate(data_loader):
            # Normalize loss by accumulation steps
            normalized_loss_scale = 1.0 / self.accumulation_steps

            with orttraining.amp.autocast():
                loss = self.training_session.train_step(inputs, labels)
                loss = loss * normalized_loss_scale

            # Scale and backward
            scaled_loss = self.scaler.scale(loss)
            scaled_loss.backward()

            # Update every accumulation_steps
            if (batch_idx + 1) % self.accumulation_steps == 0:
                # Gradient clipping before unscaling
                self.scaler.unscale_(self.training_session.get_optimizer())
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Optimizer step
                self.scaler.step(self.training_session.get_optimizer())
                self.scaler.update()
                self.training_session.lazy_reset_grad()
```

#### 2. Custom Loss Scaling Strategies

```python
class AdaptiveLossScaler:
    def __init__(self, init_scale=2**15, growth_factor=2.0, backoff_factor=0.5):
        self.scale = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = 2000
        self.consecutive_unskipped = 0

    def update_scale(self, found_inf):
        """Update loss scale based on gradient overflow"""
        if found_inf:
            # Reduce scale on overflow
            self.scale *= self.backoff_factor
            self.consecutive_unskipped = 0
        else:
            self.consecutive_unskipped += 1

            # Increase scale after growth_interval successful steps
            if self.consecutive_unskipped >= self.growth_interval:
                self.scale *= self.growth_factor
                self.consecutive_unskipped = 0

        return self.scale

    def scale_loss(self, loss):
        """Scale loss to prevent gradient underflow"""
        return loss * self.scale

    def unscale_gradients(self, gradients):
        """Unscale gradients after backward pass"""
        return [grad / self.scale for grad in gradients]
```

#### 3. Memory-Efficient Mixed Precision

```python
class MemoryEfficientAMP:
    def __init__(self, model_path):
        self.training_session = orttraining.TrainingSession(
            train_model_path=model_path,
            eval_model_path=model_path.replace('training', 'eval'),
            optimizer_model_path=model_path.replace('training', 'optimizer')
        )

        # Enable gradient checkpointing
        self.training_session.set_gradient_accumulation_steps(1)

    def memory_efficient_forward(self, inputs, labels):
        """Memory-efficient forward pass with activation checkpointing"""

        # Use activation checkpointing to save memory
        with orttraining.amp.autocast():
            # Enable gradient checkpointing for memory efficiency
            with torch.utils.checkpoint.checkpoint_sequential(
                self.training_session, segments=4):
                loss = self.training_session.train_step(inputs, labels)

        return loss
```

#### Performance Monitoring and Debugging

```python
class MixedPrecisionProfiler:
    def __init__(self):
        self.overflow_counts = 0
        self.total_steps = 0
        self.loss_scales = []

    def profile_training_step(self, trainer, inputs, labels):
        """Profile mixed precision training step"""

        # Monitor memory usage
        if torch.cuda.is_available():
            memory_before = torch.cuda.memory_allocated()

        # Training step with profiling
        start_time = time.time()
        loss = trainer.train_step_with_amp(inputs, labels)
        step_time = time.time() - start_time

        # Monitor memory usage
        if torch.cuda.is_available():
            memory_after = torch.cuda.memory_allocated()
            memory_delta = memory_after - memory_before

        # Check for gradient overflow
        if hasattr(trainer.scaler, '_found_inf_per_device'):
            if trainer.scaler._found_inf_per_device(torch.device('cuda:0')):
                self.overflow_counts += 1

        self.total_steps += 1
        self.loss_scales.append(trainer.scaler.get_scale())

        # Log statistics
        if self.total_steps % 100 == 0:
            overflow_rate = self.overflow_counts / self.total_steps
            avg_scale = np.mean(self.loss_scales[-100:])

            print(f"Step {self.total_steps}:")
            print(f"  Loss: {loss:.6f}")
            print(f"  Step time: {step_time:.3f}s")
            print(f"  Memory delta: {memory_delta / 1024**2:.2f} MB")
            print(f"  Overflow rate: {overflow_rate:.4f}")
            print(f"  Current loss scale: {avg_scale:.0f}")

        return {
            'loss': loss,
            'step_time': step_time,
            'memory_delta': memory_delta,
            'overflow_detected': self.overflow_counts > 0
        }
```

#### Real-World Mixed-Precision Training Examples

#### Example 1: GPT-2 Training with Mixed Precision
```python
import onnxruntime.training as orttraining
import torch
import numpy as np
from transformers import GPT2TokenizerFast

class GPT2MixedPrecisionTrainer:
    def __init__(self, model_path, vocab_size=50257, max_length=1024):
        self.model_path = model_path
        self.vocab_size = vocab_size
        self.max_length = max_length

        # Initialize training session
        self.training_session = orttraining.TrainingSession(
            train_model_path=model_path,
            eval_model_path=model_path.replace('training', 'eval'),
            optimizer_model_path=model_path.replace('training', 'optimizer'),
            device_id=0
        )

        # Advanced gradient scaler with custom parameters
        self.scaler = orttraining.amp.GradScaler(
            init_scale=2**16,           # Higher initial scale for stability
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=2000        # Conservative growth
        )

        # Tokenizer for text processing
        self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Training statistics
        self.training_stats = {
            'total_steps': 0,
            'overflow_steps': 0,
            'loss_history': [],
            'gradient_norms': []
        }

    def prepare_batch(self, texts, max_length=None):
        """Prepare a batch of text for training"""
        if max_length is None:
            max_length = self.max_length

        # Tokenize texts
        encoded = self.tokenizer(
            texts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors='np'
        )

        input_ids = encoded['input_ids'].astype(np.int64)
        attention_mask = encoded['attention_mask'].astype(np.int64)

        # For language modeling, labels are the same as input_ids
        labels = input_ids.copy()

        # Shift labels for next token prediction
        labels = np.roll(labels, -1, axis=1)
        labels[:, -1] = -100  # Ignore last token in loss computation

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

    def train_step_with_gradient_accumulation(self, batch, accumulation_steps=4):
        """Training step with gradient accumulation and mixed precision"""

        total_loss = 0

        for micro_step in range(accumulation_steps):
            # Get micro-batch
            batch_size = len(batch['input_ids'])
            micro_batch_size = batch_size // accumulation_steps
            start_idx = micro_step * micro_batch_size
            end_idx = start_idx + micro_batch_size

            micro_batch = {
                'input_ids': batch['input_ids'][start_idx:end_idx],
                'attention_mask': batch['attention_mask'][start_idx:end_idx],
                'labels': batch['labels'][start_idx:end_idx]
            }

            # Forward pass with autocast
            with orttraining.amp.autocast():
                loss = self.training_session.train_step(micro_batch['input_ids'],
                                                       micro_batch['attention_mask'],
                                                       micro_batch['labels'])

                # Scale loss by accumulation steps
                loss = loss / accumulation_steps

            # Scale and backward
            scaled_loss = self.scaler.scale(loss)
            scaled_loss.backward()

            total_loss += loss.item()

        # Gradient clipping before optimizer step
        self.scaler.unscale_(self.training_session.get_optimizer())

        # Compute gradient norm for monitoring
        total_norm = 0
        for param in self.training_session.get_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(
            self.training_session.get_parameters(),
            max_norm=1.0
        )

        # Optimizer step
        self.scaler.step(self.training_session.get_optimizer())
        self.scaler.update()

        # Reset gradients
        self.training_session.lazy_reset_grad()

        # Update statistics
        self.training_stats['total_steps'] += 1
        self.training_stats['loss_history'].append(total_loss)
        self.training_stats['gradient_norms'].append(total_norm)

        # Check for overflow
        if self.scaler._found_inf_per_device(torch.device('cuda:0')):
            self.training_stats['overflow_steps'] += 1

        return total_loss, total_norm

    def adaptive_loss_scaling(self):
        """Implement adaptive loss scaling based on training dynamics"""
        recent_overflows = self.training_stats['overflow_steps']
        total_steps = self.training_stats['total_steps']

        if total_steps > 1000:  # Only adapt after warmup
            overflow_rate = recent_overflows / total_steps

            if overflow_rate > 0.05:  # Too many overflows
                self.scaler._scale = max(self.scaler._scale * 0.5, 2**8)
                print(f"Reduced loss scale to {self.scaler._scale}")
            elif overflow_rate < 0.01:  # Very stable, can increase
                self.scaler._scale = min(self.scaler._scale * 1.1, 2**20)
                print(f"Increased loss scale to {self.scaler._scale}")

    def get_training_metrics(self):
        """Get comprehensive training metrics"""
        total_steps = self.training_stats['total_steps']
        if total_steps == 0:
            return {}

        recent_losses = self.training_stats['loss_history'][-100:]
        recent_grad_norms = self.training_stats['gradient_norms'][-100:]

        metrics = {
            'total_steps': total_steps,
            'overflow_rate': self.training_stats['overflow_steps'] / total_steps,
            'current_loss_scale': float(self.scaler._scale),
            'avg_loss_recent': np.mean(recent_losses) if recent_losses else 0,
            'avg_grad_norm_recent': np.mean(recent_grad_norms) if recent_grad_norms else 0,
            'loss_std_recent': np.std(recent_losses) if recent_losses else 0
        }

        return metrics

# Complete training loop example
def gpt2_mixed_precision_training():
    """Complete example of GPT-2 training with mixed precision"""

    trainer = GPT2MixedPrecisionTrainer("gpt2_training_model.onnx")

    # Sample training data (replace with your dataset)
    training_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming the world.",
        "ONNX Runtime provides high-performance inference.",
        # Add more training texts...
    ]

    batch_size = 8
    accumulation_steps = 4
    num_epochs = 3

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Shuffle training data
        np.random.shuffle(training_texts)

        # Process in batches
        for i in range(0, len(training_texts), batch_size):
            batch_texts = training_texts[i:i + batch_size]
            batch = trainer.prepare_batch(batch_texts)

            # Training step
            loss, grad_norm = trainer.train_step_with_gradient_accumulation(
                batch, accumulation_steps)

            # Adaptive loss scaling
            if trainer.training_stats['total_steps'] % 500 == 0:
                trainer.adaptive_loss_scaling()

            # Logging
            if trainer.training_stats['total_steps'] % 100 == 0:
                metrics = trainer.get_training_metrics()
                print(f"Step {metrics['total_steps']}: "
                      f"Loss={metrics['avg_loss_recent']:.4f}, "
                      f"GradNorm={metrics['avg_grad_norm_recent']:.4f}, "
                      f"OverflowRate={metrics['overflow_rate']:.4f}, "
                      f"LossScale={metrics['current_loss_scale']:.0f}")

    return trainer
```

#### Example 2: Vision Transformer Training with Mixed Precision
```python
class ViTMixedPrecisionTrainer:
    def __init__(self, model_path, image_size=224, patch_size=16):
        self.model_path = model_path
        self.image_size = image_size
        self.patch_size = patch_size

        # Initialize training session with specific optimizations for ViT
        self.training_session = orttraining.TrainingSession(
            train_model_path=model_path,
            eval_model_path=model_path.replace('training', 'eval'),
            optimizer_model_path=model_path.replace('training', 'optimizer')
        )

        # Custom scaler for vision tasks
        self.scaler = orttraining.amp.GradScaler(
            init_scale=2**15,       # Lower initial scale for vision
            growth_factor=2.0,
            backoff_factor=0.75,    # More conservative backoff
            growth_interval=1500    # Faster adaptation
        )

        # Layer-wise learning rate scaling for ViT
        self.layer_lr_decay = 0.85  # Decay factor for deeper layers

    def prepare_image_batch(self, images, labels):
        """Prepare batch of images with proper preprocessing"""
        # Normalize images (ImageNet stats)
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)

        normalized_images = (images - mean) / std

        return {
            'pixel_values': normalized_images.astype(np.float32),
            'labels': labels.astype(np.int64)
        }

    def train_step_with_layer_lr_scaling(self, batch):
        """Training step with layer-wise learning rate scaling"""

        # Forward pass with autocast
        with orttraining.amp.autocast():
            logits = self.training_session.forward(batch['pixel_values'])

            # Compute cross-entropy loss
            loss = self.compute_cross_entropy_loss(logits, batch['labels'])

        # Scale loss
        scaled_loss = self.scaler.scale(loss)

        # Backward pass
        scaled_loss.backward()

        # Apply layer-wise learning rate scaling
        self.apply_layer_wise_lr_scaling()

        # Gradient clipping
        self.scaler.unscale_(self.training_session.get_optimizer())
        torch.nn.utils.clip_grad_norm_(
            self.training_session.get_parameters(),
            max_norm=5.0  # Higher clip for vision models
        )

        # Optimizer step
        self.scaler.step(self.training_session.get_optimizer())
        self.scaler.update()

        # Reset gradients
        self.training_session.lazy_reset_grad()

        return loss.item()

    def apply_layer_wise_lr_scaling(self):
        """Apply different learning rates to different layers"""
        layer_count = 0

        for name, param in self.training_session.named_parameters():
            if param.grad is not None:
                # Determine layer depth
                if 'embeddings' in name:
                    layer_depth = 0
                elif 'encoder.layer' in name:
                    # Extract layer number
                    layer_num = int(name.split('encoder.layer.')[1].split('.')[0])
                    layer_depth = layer_num + 1
                elif 'classifier' in name:
                    layer_depth = 12  # Assume 12 layer ViT
                else:
                    layer_depth = 6   # Default middle layer

                # Apply learning rate decay
                lr_scale = self.layer_lr_decay ** (12 - layer_depth)
                param.grad *= lr_scale

    def compute_cross_entropy_loss(self, logits, labels):
        """Compute cross-entropy loss compatible with mixed precision"""
        # Convert to torch tensors if needed
        if isinstance(logits, np.ndarray):
            logits = torch.from_numpy(logits)
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)

        # Use label smoothing for better generalization
        return torch.nn.functional.cross_entropy(
            logits, labels,
            label_smoothing=0.1
        )

    def warmup_lr_schedule(self, step, warmup_steps=1000, base_lr=1e-4):
        """Implement learning rate warmup schedule"""
        if step < warmup_steps:
            lr = base_lr * (step / warmup_steps)
        else:
            # Cosine decay after warmup
            progress = (step - warmup_steps) / (10000 - warmup_steps)  # Assume 10k total steps
            lr = base_lr * 0.5 * (1 + np.cos(np.pi * progress))

        # Update optimizer learning rate
        for param_group in self.training_session.get_optimizer().param_groups:
            param_group['lr'] = lr

        return lr

# Advanced mixed precision techniques
class AdvancedMixedPrecisionOptimizer:
    def __init__(self):
        self.loss_scale_history = []
        self.gradient_statistics = {}

    def dynamic_precision_selection(self, model, layer_sensitivities):
        """Dynamically select precision for each layer based on sensitivity"""
        precision_map = {}

        for layer_name, sensitivity in layer_sensitivities.items():
            if sensitivity > 0.1:  # High sensitivity
                precision_map[layer_name] = torch.float32
            elif sensitivity > 0.05:  # Medium sensitivity
                precision_map[layer_name] = torch.bfloat16
            else:  # Low sensitivity
                precision_map[layer_name] = torch.float16

        return precision_map

    def gradient_overflow_prediction(self, gradient_norms, window_size=100):
        """Predict gradient overflow before it happens"""
        if len(gradient_norms) < window_size:
            return False

        recent_norms = gradient_norms[-window_size:]
        trend = np.polyfit(range(window_size), recent_norms, 1)[0]

        # Predict overflow if gradient norm is growing rapidly
        predicted_norm = recent_norms[-1] + trend * 10
        overflow_threshold = 1000.0  # Empirical threshold

        return predicted_norm > overflow_threshold

    def adaptive_gradient_clipping(self, model, percentile=90):
        """Implement adaptive gradient clipping based on gradient statistics"""
        all_grad_norms = []

        for param in model.parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                all_grad_norms.append(grad_norm)

        if all_grad_norms:
            # Use percentile-based clipping
            clip_value = np.percentile(all_grad_norms, percentile)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

            return clip_value

        return 0.0

# Usage example combining all techniques
def advanced_mixed_precision_training():
    """Demonstrate advanced mixed precision training techniques"""

    # Initialize trainers
    gpt2_trainer = GPT2MixedPrecisionTrainer("gpt2_model.onnx")
    vit_trainer = ViTMixedPrecisionTrainer("vit_model.onnx")
    optimizer = AdvancedMixedPrecisionOptimizer()

    # Training loop with advanced techniques
    print("Starting advanced mixed precision training...")

    # GPT-2 training
    gpt2_trainer = gpt2_mixed_precision_training()

    # Print final statistics
    final_metrics = gpt2_trainer.get_training_metrics()
    print(f"\nFinal Training Metrics:")
    for key, value in final_metrics.items():
        print(f"{key}: {value}")
```### Debugging and Profiling

#### Advanced Debugging:
```cpp
// Enable verbose logging
#include "core/common/logging/logging.h"

// Set log level
onnxruntime::logging::LoggingManager::DefaultLogger().SetDefaultLoggerSeverity(
    onnxruntime::logging::Severity::kVERBOSE);

// Custom logger
class CustomLogger : public onnxruntime::logging::ISink {
 public:
  void SendImpl(const onnxruntime::logging::Timestamp& timestamp,
                const std::string& logger_id,
                const onnxruntime::logging::Capture& message) override {
    // Custom logging implementation
    std::cout << "[" << logger_id << "] " << message.str() << std::endl;
  }
};
```

#### Performance Profiling:
```python
# Detailed profiling
options = ort.SessionOptions()
options.enable_profiling = True
options.profile_file_prefix = "ort_profile"

session = ort.InferenceSession("model.onnx", options)

# Run inference
session.run(None, inputs)

# End profiling and analyze results
prof_file = session.end_profiling()

# Parse profiling data
import json
with open(prof_file, 'r') as f:
    profile_data = json.load(f)

for event in profile_data['traceEvents']:
    if event['ph'] == 'X':  # Complete events
        print(f"Op: {event['name']}, Duration: {event['dur']}μs")
```

---

## Chapter 11: Advanced Optimization Theories and Techniques

### Graph Fusion Patterns and Algorithms

Graph fusion combines multiple operations into single, optimized kernels to reduce memory bandwidth and improve performance.

#### Common Fusion Patterns

#### 1. Elementwise Fusion
```cpp
// Fuse: Add → Relu → Mul into single kernel
class ElementwiseFusionPattern : public GraphTransformer {
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level) const override {
    for (auto& node : graph.Nodes()) {
      if (auto fusion_candidate = FindElementwiseFusionCandidate(node)) {
        CreateFusedElementwiseKernel(graph, fusion_candidate);
        modified = true;
      }
    }
    return Status::OK();
  }

 private:
  struct FusionCandidate {
    std::vector<Node*> nodes_to_fuse;
    std::string fused_kernel_name;
    std::vector<float> parameters;
  };

  std::unique_ptr<FusionCandidate> FindElementwiseFusionCandidate(Node& node) {
    if (node.OpType() != "Add") return nullptr;

    auto next_node = GetSingleConsumer(node);
    if (!next_node || next_node->OpType() != "Relu") return nullptr;

    auto final_node = GetSingleConsumer(*next_node);
    if (!final_node || final_node->OpType() != "Mul") return nullptr;

    // Create fusion candidate
    auto candidate = std::make_unique<FusionCandidate>();
    candidate->nodes_to_fuse = {&node, next_node, final_node};
    candidate->fused_kernel_name = "AddReluMul";

    return candidate;
  }
};
```

#### 2. Convolution Fusion Patterns
```cpp
// Conv + BatchNorm + ReLU fusion
class ConvBNReluFusion : public GraphTransformer {
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level) const override {
    auto conv_nodes = graph.GetNodesOfOpType("Conv");

    for (auto* conv_node : conv_nodes) {
      if (auto fusion_opportunity = AnalyzeConvFusion(*conv_node)) {
        // Mathematically fuse BatchNorm into Conv weights/bias
        FuseConvBatchNorm(graph, fusion_opportunity);
        modified = true;
      }
    }

    return Status::OK();
  }

 private:
  void FuseConvBatchNorm(Graph& graph, const ConvBNFusionInfo& info) {
    // Get BatchNorm parameters: scale, bias, mean, variance
    auto bn_scale = GetConstantTensor(info.bn_node, "scale");
    auto bn_bias = GetConstantTensor(info.bn_node, "B");
    auto bn_mean = GetConstantTensor(info.bn_node, "mean");
    auto bn_var = GetConstantTensor(info.bn_node, "var");

    // Get Conv weights and bias
    auto conv_weights = GetConstantTensor(info.conv_node, "W");
    auto conv_bias = GetConstantTensor(info.conv_node, "B");

    // Fuse: new_weight = weight * (scale / sqrt(var + epsilon))
    //       new_bias = (bias - mean) * (scale / sqrt(var + epsilon)) + bn_bias
    auto fused_weights = FuseBatchNormIntoConvWeights(
        conv_weights, bn_scale, bn_var, info.epsilon);
    auto fused_bias = FuseBatchNormIntoConvBias(
        conv_bias, bn_mean, bn_scale, bn_var, bn_bias, info.epsilon);

    // Update Conv node with fused parameters
    UpdateNodeInput(graph, *info.conv_node, 1, fused_weights);
    UpdateNodeInput(graph, *info.conv_node, 2, fused_bias);

    // Remove BatchNorm node
    graph.RemoveNode(info.bn_node->Index());
  }
};
```

#### 3. Attention Fusion
```cpp
// Multi-Head Attention fusion pattern
class MultiHeadAttentionFusion : public GraphTransformer {
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level) const override {
    auto attention_patterns = FindAttentionPatterns(graph);

    for (const auto& pattern : attention_patterns) {
      if (CanFuseAttentionPattern(pattern)) {
        CreateFusedAttentionNode(graph, pattern);
        modified = true;
      }
    }

    return Status::OK();
  }

 private:
  struct AttentionPattern {
    Node* q_linear;     // Query projection
    Node* k_linear;     // Key projection
    Node* v_linear;     // Value projection
    Node* qk_matmul;    // Q @ K^T
    Node* scale_div;    // Scale by sqrt(d_k)
    Node* softmax;      // Attention weights
    Node* av_matmul;    // Attention @ V
    Node* output_proj;  // Output projection

    int64_t num_heads;
    int64_t head_dim;
    float scale_factor;
  };

  void CreateFusedAttentionNode(Graph& graph, const AttentionPattern& pattern) {
    // Create optimized attention kernel that computes:
    // Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V

    auto fused_attention = graph.AddNode(
        "FusedMultiHeadAttention",
        {pattern.q_linear->MutableInputDefs()[0]},  // Input
        {pattern.output_proj->MutableOutputDefs()[0]},  // Output
        "custom_domain"
    );

    // Add attributes for fused attention
    fused_attention->AddAttribute("num_heads", pattern.num_heads);
    fused_attention->AddAttribute("head_dim", pattern.head_dim);
    fused_attention->AddAttribute("scale", pattern.scale_factor);

    // Remove original attention pattern nodes
    RemoveAttentionPatternNodes(graph, pattern);
  }
};
```

### Memory Optimization Algorithms

#### Memory Layout Optimization
```cpp
class MemoryLayoutOptimizer : public GraphTransformer {
 public:
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level) const override {
    // Analyze data layout requirements
    auto layout_analysis = AnalyzeDataLayouts(graph);

    // Apply layout transformations
    for (const auto& transformation : layout_analysis.transformations) {
      if (transformation.benefit_score > cost_threshold_) {
        ApplyLayoutTransformation(graph, transformation);
        modified = true;
      }
    }

    return Status::OK();
  }

 private:
  struct LayoutTransformation {
    Node* target_node;
    std::string current_layout;  // e.g., "NCHW"
    std::string target_layout;   // e.g., "NHWC"
    float benefit_score;         // Performance improvement estimate
    std::vector<Node*> transpose_nodes_to_insert;
  };

  LayoutAnalysis AnalyzeDataLayouts(const Graph& graph) {
    LayoutAnalysis analysis;

    // Track data layout through the graph
    std::unordered_map<const NodeArg*, std::string> layout_map;

    for (const auto& node : graph.Nodes()) {
      auto preferred_layout = GetPreferredLayout(node);
      auto current_layout = GetCurrentLayout(node, layout_map);

      if (preferred_layout != current_layout) {
        // Calculate benefit of layout transformation
        float benefit = CalculateLayoutBenefit(node, current_layout, preferred_layout);

        if (benefit > 0) {
          LayoutTransformation transform;
          transform.target_node = const_cast<Node*>(&node);
          transform.current_layout = current_layout;
          transform.target_layout = preferred_layout;
          transform.benefit_score = benefit;

          analysis.transformations.push_back(transform);
        }
      }
    }

    return analysis;
  }
};
```

#### Memory Pool Management
```cpp
class AdvancedMemoryPool {
 public:
  AdvancedMemoryPool(size_t initial_size, IAllocator* base_allocator)
      : base_allocator_(base_allocator), pool_size_(initial_size) {
    // Pre-allocate memory pool
    pool_memory_ = base_allocator_->Alloc(pool_size_);

    // Initialize free block list
    FreeBlock initial_block{0, pool_size_};
    free_blocks_.insert(initial_block);
  }

  void* Alloc(size_t size) override {
    size = AlignSize(size, alignment_);

    // Find best-fit free block
    auto best_fit = FindBestFitBlock(size);
    if (best_fit == free_blocks_.end()) {
      // Pool exhausted, expand or fallback to base allocator
      return ExpandPoolOrFallback(size);
    }

    // Allocate from free block
    void* ptr = static_cast<char*>(pool_memory_) + best_fit->offset;

    // Update free block list
    if (best_fit->size > size) {
      // Split block
      FreeBlock remaining{best_fit->offset + size, best_fit->size - size};
      free_blocks_.erase(best_fit);
      free_blocks_.insert(remaining);
    } else {
      free_blocks_.erase(best_fit);
    }

    // Track allocation
    allocated_blocks_[ptr] = {best_fit->offset, size};

    return ptr;
  }

  void Free(void* ptr) override {
    auto it = allocated_blocks_.find(ptr);
    if (it == allocated_blocks_.end()) {
      // Not from pool, delegate to base allocator
      base_allocator_->Free(ptr);
      return;
    }

    // Add back to free list
    FreeBlock freed_block{it->second.offset, it->second.size};
    allocated_blocks_.erase(it);

    // Coalesce adjacent free blocks
    CoalesceFreeBlocks(freed_block);
  }

 private:
  struct FreeBlock {
    size_t offset;
    size_t size;

    bool operator<(const FreeBlock& other) const {
      return size < other.size;  // Order by size for best-fit
    }
  };

  void CoalesceFreeBlocks(FreeBlock new_block) {
    // Find adjacent blocks and merge
    auto it = free_blocks_.begin();
    while (it != free_blocks_.end()) {
      if (it->offset + it->size == new_block.offset) {
        // Merge with previous block
        new_block.offset = it->offset;
        new_block.size += it->size;
        it = free_blocks_.erase(it);
      } else if (new_block.offset + new_block.size == it->offset) {
        // Merge with next block
        new_block.size += it->size;
        it = free_blocks_.erase(it);
      } else {
        ++it;
      }
    }

    free_blocks_.insert(new_block);
  }
};
```

### Automatic Kernel Tuning

#### Auto-Tuning Framework
```cpp
class AutoTuner {
 public:
  struct KernelConfig {
    std::unordered_map<std::string, int> int_params;
    std::unordered_map<std::string, float> float_params;
    std::string kernel_name;
    float performance_score = 0.0f;
  };

  KernelConfig TuneKernel(const OpKernelInfo& kernel_info,
                         const std::vector<TensorShape>& input_shapes) {
    auto search_space = GenerateSearchSpace(kernel_info, input_shapes);
    auto best_config = BayesianOptimization(search_space, kernel_info);

    return best_config;
  }

 private:
  std::vector<KernelConfig> GenerateSearchSpace(
      const OpKernelInfo& kernel_info,
      const std::vector<TensorShape>& input_shapes) {

    std::vector<KernelConfig> configs;

    if (kernel_info.GetKernelDef().OpType() == "Conv") {
      // Convolution-specific tuning parameters
      for (int tile_size : {8, 16, 32, 64}) {
        for (int block_size : {128, 256, 512}) {
          for (bool use_winograd : {true, false}) {
            KernelConfig config;
            config.int_params["tile_size"] = tile_size;
            config.int_params["block_size"] = block_size;
            config.int_params["use_winograd"] = use_winograd ? 1 : 0;
            config.kernel_name = "OptimizedConv";
            configs.push_back(config);
          }
        }
      }
    } else if (kernel_info.GetKernelDef().OpType() == "MatMul") {
      // Matrix multiplication tuning parameters
      for (int m_tile : {32, 64, 128}) {
        for (int n_tile : {32, 64, 128}) {
          for (int k_tile : {16, 32, 64}) {
            KernelConfig config;
            config.int_params["m_tile"] = m_tile;
            config.int_params["n_tile"] = n_tile;
            config.int_params["k_tile"] = k_tile;
            config.kernel_name = "TiledMatMul";
            configs.push_back(config);
          }
        }
      }
    }

    return configs;
  }

  KernelConfig BayesianOptimization(
      const std::vector<KernelConfig>& search_space,
      const OpKernelInfo& kernel_info) {

    // Gaussian Process surrogate model
    GaussianProcess gp_model;

    // Sample initial configurations
    std::vector<KernelConfig> evaluated_configs;
    for (int i = 0; i < std::min(10, static_cast<int>(search_space.size())); ++i) {
      auto config = search_space[i];
      config.performance_score = EvaluateKernelPerformance(config, kernel_info);
      evaluated_configs.push_back(config);

      // Update surrogate model
      gp_model.AddObservation(config, config.performance_score);
    }

    // Bayesian optimization loop
    for (int iteration = 0; iteration < max_iterations_; ++iteration) {
      // Acquisition function (Expected Improvement)
      auto next_config = AcquisitionFunction(search_space, gp_model);

      // Evaluate new configuration
      next_config.performance_score = EvaluateKernelPerformance(next_config, kernel_info);
      evaluated_configs.push_back(next_config);

      // Update surrogate model
      gp_model.AddObservation(next_config, next_config.performance_score);
    }

    // Return best configuration
    return *std::max_element(evaluated_configs.begin(), evaluated_configs.end(),
        [](const KernelConfig& a, const KernelConfig& b) {
          return a.performance_score < b.performance_score;
        });
  }

  float EvaluateKernelPerformance(const KernelConfig& config,
                                 const OpKernelInfo& kernel_info) {
    // Create kernel instance with configuration
    auto kernel = CreateTunedKernel(config, kernel_info);

    // Benchmark kernel performance
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < benchmark_iterations_; ++i) {
      kernel->Compute(benchmark_context_);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time).count();

    // Return inverse of execution time (higher is better)
    return 1.0f / (duration / static_cast<float>(benchmark_iterations_));
  }
};
```

#### Real-World Advanced Optimization Examples

#### Example 1: Automatic Operator Fusion for Custom Workloads
```cpp
class WorkloadSpecificFusionOptimizer {
 public:
  // Analyze workload patterns and create custom fusion rules
  struct FusionRule {
    std::vector<std::string> pattern;  // Sequence of operations
    std::string fused_op_name;         // Name of fused operation
    float expected_speedup;            // Expected performance improvement
    std::function<bool(const std::vector<Node*>&)> validator;
  };

  WorkloadSpecificFusionOptimizer() {
    // Define workload-specific fusion patterns
    InitializeCustomFusionRules();
  }

  void OptimizeForWorkload(Graph& graph, const WorkloadProfile& profile) {
    // Analyze graph for hot paths
    auto hot_paths = AnalyzeHotPaths(graph, profile);

    // Apply workload-specific optimizations
    for (const auto& path : hot_paths) {
      OptimizeHotPath(graph, path);
    }
  }

 private:
  std::vector<FusionRule> fusion_rules_;

  void InitializeCustomFusionRules() {
    // Computer vision workload: Conv + BN + ReLU + Pool
    fusion_rules_.push_back({
      {"Conv", "BatchNormalization", "Relu", "MaxPool"},
      "ConvBNReluPool",
      2.5f,  // Expected 2.5x speedup
      [](const std::vector<Node*>& nodes) {
        // Validate that fusion is beneficial
        return ValidateConvBNReluPoolFusion(nodes);
      }
    });

    // NLP workload: MatMul + Add + LayerNorm
    fusion_rules_.push_back({
      {"MatMul", "Add", "LayerNormalization"},
      "FusedLinearLayerNorm",
      1.8f,
      [](const std::vector<Node*>& nodes) {
        return ValidateLinearLayerNormFusion(nodes);
      }
    });

    // Attention pattern: Q*K^T + Scale + Softmax + Dropout + *V
    fusion_rules_.push_back({
      {"MatMul", "Div", "Softmax", "Dropout", "MatMul"},
      "FusedAttention",
      4.0f,  // Significant speedup for attention
      [](const std::vector<Node*>& nodes) {
        return ValidateAttentionFusion(nodes);
      }
    });
  }

  struct WorkloadProfile {
    std::unordered_map<std::string, float> op_frequencies;
    std::unordered_map<std::string, float> op_latencies;
    std::vector<std::vector<std::string>> common_patterns;
  };

  std::vector<std::vector<Node*>> AnalyzeHotPaths(
      const Graph& graph, const WorkloadProfile& profile) {
    std::vector<std::vector<Node*>> hot_paths;

    // Find frequently executed operation sequences
    for (const auto& pattern : profile.common_patterns) {
      auto paths = FindPatternInGraph(graph, pattern);

      // Filter by execution frequency and latency
      for (const auto& path : paths) {
        float total_latency = CalculatePathLatency(path, profile);
        float execution_frequency = CalculatePathFrequency(path, profile);

        if (total_latency * execution_frequency > threshold_) {
          hot_paths.push_back(path);
        }
      }
    }

    return hot_paths;
  }
};

// Example usage
void optimize_resnet_workload() {
  // Load ResNet model
  auto model = onnx::load("resnet50.onnx");
  Graph graph(model.graph());

  // Profile workload
  WorkloadSpecificFusionOptimizer::WorkloadProfile profile;
  profile.op_frequencies = {
    {"Conv", 0.4f}, {"BatchNormalization", 0.25f},
    {"Relu", 0.2f}, {"MaxPool", 0.1f}, {"Add", 0.05f}
  };

  // Apply workload-specific optimizations
  WorkloadSpecificFusionOptimizer optimizer;
  optimizer.OptimizeForWorkload(graph, profile);

  // Save optimized model
  onnx::save(model, "resnet50_optimized.onnx");
}
```

#### Example 2: Advanced Memory Optimization with Prefetching
```cpp
class AdvancedMemoryOptimizer {
 public:
  struct MemoryAccessPattern {
    std::vector<size_t> access_sequence;
    std::vector<size_t> tensor_sizes;
    std::vector<float> access_probabilities;
  };

  class PrefetchingAllocator : public IAllocator {
   public:
    PrefetchingAllocator(IAllocator* base_allocator,
                        const MemoryAccessPattern& pattern)
        : base_allocator_(base_allocator), access_pattern_(pattern) {
      InitializePrefetchQueue();
    }

    void* Alloc(size_t size) override {
      // Check if this allocation was prefetched
      auto prefetched = prefetch_cache_.find(size);
      if (prefetched != prefetch_cache_.end() && !prefetched->second.empty()) {
        void* ptr = prefetched->second.front();
        prefetched->second.pop();
        return ptr;
      }

      // Trigger prefetching for future allocations
      TriggerPrefetch(size);

      return base_allocator_->Alloc(size);
    }

    void Free(void* ptr) override {
      // Instead of immediately freeing, add to prefetch cache
      size_t size = GetAllocationSize(ptr);

      if (ShouldCachePrefetch(size)) {
        prefetch_cache_[size].push(ptr);
      } else {
        base_allocator_->Free(ptr);
      }
    }

   private:
    IAllocator* base_allocator_;
    MemoryAccessPattern access_pattern_;
    std::unordered_map<size_t, std::queue<void*>> prefetch_cache_;

    void InitializePrefetchQueue() {
      // Pre-allocate frequently used tensor sizes
      for (size_t i = 0; i < access_pattern_.tensor_sizes.size(); ++i) {
        size_t size = access_pattern_.tensor_sizes[i];
        float probability = access_pattern_.access_probabilities[i];

        if (probability > 0.7f) {  // High probability
          // Pre-allocate multiple buffers
          int prefetch_count = static_cast<int>(probability * 5);
          for (int j = 0; j < prefetch_count; ++j) {
            void* ptr = base_allocator_->Alloc(size);
            prefetch_cache_[size].push(ptr);
          }
        }
      }
    }

    void TriggerPrefetch(size_t requested_size) {
      // Predict next allocations based on patterns
      auto next_sizes = PredictNextAllocations(requested_size);

      for (size_t next_size : next_sizes) {
        if (prefetch_cache_[next_size].size() < max_prefetch_count_) {
          void* ptr = base_allocator_->Alloc(next_size);
          prefetch_cache_[next_size].push(ptr);
        }
      }
    }
  };

  // NUMA-aware memory optimization
  class NUMAOptimizedAllocator : public IAllocator {
   public:
    NUMAOptimizedAllocator() {
      InitializeNUMATopology();
    }

    void* Alloc(size_t size) override {
      // Determine optimal NUMA node based on current thread
      int current_thread = GetCurrentThreadId();
      int optimal_node = GetOptimalNUMANode(current_thread);

      // Allocate on specific NUMA node
      return numa_alloc_onnode(size, optimal_node);
    }

    void Free(void* ptr) override {
      numa_free(ptr, GetAllocationSize(ptr));
    }

   private:
    std::vector<int> numa_nodes_;
    std::unordered_map<int, int> thread_to_numa_map_;

    void InitializeNUMATopology() {
      int num_nodes = numa_max_node() + 1;
      numa_nodes_.resize(num_nodes);

      // Map CPU cores to NUMA nodes
      for (int cpu = 0; cpu < numa_num_configured_cpus(); ++cpu) {
        int node = numa_node_of_cpu(cpu);
        thread_to_numa_map_[cpu] = node;
      }
    }
  };
};

// Example: Memory layout optimization for transformer models
class TransformerMemoryOptimizer {
 public:
  void OptimizeTransformerMemoryLayout(Graph& graph) {
    // Identify attention blocks
    auto attention_blocks = FindAttentionBlocks(graph);

    for (auto& block : attention_blocks) {
      OptimizeAttentionMemoryLayout(block);
    }

    // Optimize feed-forward blocks
    auto ffn_blocks = FindFFNBlocks(graph);
    for (auto& block : ffn_blocks) {
      OptimizeFFNMemoryLayout(block);
    }
  }

 private:
  void OptimizeAttentionMemoryLayout(AttentionBlock& block) {
    // Reorganize Q, K, V matrices for better cache locality
    // Layout: [batch, num_heads, seq_len, head_dim] ->
    //         [batch, seq_len, num_heads, head_dim]

    for (auto* qkv_node : {block.q_proj, block.k_proj, block.v_proj}) {
      InsertTransposeNode(qkv_node, {0, 2, 1, 3});
    }

    // Fuse attention computation to minimize memory movement
    CreateFusedAttentionNode(block);
  }

  void OptimizeFFNMemoryLayout(FFNBlock& block) {
    // Use activation checkpointing for large FFN layers
    if (GetTensorSize(block.intermediate) > large_tensor_threshold_) {
      EnableGradientCheckpointing(block.up_proj);
      EnableGradientCheckpointing(block.down_proj);
    }

    // Apply weight matrix layout optimization
    OptimizeWeightMatrixLayout(block.up_proj);
    OptimizeWeightMatrixLayout(block.down_proj);
  }
};
```

#### Example 3: Dynamic Model Optimization at Runtime
```python
class RuntimeModelOptimizer:
    def __init__(self, base_model_path):
        self.base_model_path = base_model_path
        self.optimization_history = []
        self.performance_baseline = None

    def adaptive_optimization(self, workload_characteristics):
        """Dynamically optimize model based on runtime workload"""

        # Analyze current workload
        batch_size_pattern = workload_characteristics['batch_sizes']
        input_shape_pattern = workload_characteristics['input_shapes']
        latency_requirements = workload_characteristics['latency_sla']

        # Choose optimization strategy
        if self._is_throughput_critical(workload_characteristics):
            return self._optimize_for_throughput(batch_size_pattern)
        elif self._is_latency_critical(workload_characteristics):
            return self._optimize_for_latency(latency_requirements)
        else:
            return self._balanced_optimization(workload_characteristics)

    def _optimize_for_throughput(self, batch_patterns):
        """Optimize for maximum throughput"""
        # Use largest common batch size for optimization
        optimal_batch_size = max(set(batch_patterns), key=batch_patterns.count)

        # Apply batch-specific optimizations
        optimized_model = self._create_batch_optimized_model(optimal_batch_size)

        # Enable aggressive fusion
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.add_session_config_entry("optimization.enable_gelu_approximation", "1")
        session_options.add_session_config_entry("optimization.enable_attention_fusion", "1")

        return ort.InferenceSession(optimized_model, session_options)

    def _optimize_for_latency(self, latency_sla):
        """Optimize for minimum latency"""
        # Use quantization if latency is very strict
        if latency_sla < 10:  # ms
            quantized_model = self._apply_aggressive_quantization()
            model_path = quantized_model
        else:
            model_path = self.base_model_path

        # Configure for low latency
        session_options = ort.SessionOptions()
        session_options.inter_op_num_threads = 1  # Minimize context switching
        session_options.intra_op_num_threads = min(4, os.cpu_count())
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        return ort.InferenceSession(model_path, session_options)

    def _apply_aggressive_quantization(self):
        """Apply quantization optimized for latency"""
        # Dynamic quantization with optimized settings
        quantized_path = "temp_quantized_model.onnx"

        quantize_dynamic(
            model_input=self.base_model_path,
            model_output=quantized_path,
            weight_type=QuantType.QUInt8,
            per_channel=False,  # Faster but less accurate
            reduce_range=True,  # Better for CPU
            optimize_model=True
        )

        return quantized_path

    def continuous_optimization(self, performance_monitor):
        """Continuously optimize based on performance feedback"""
        current_metrics = performance_monitor.get_metrics()

        if self.performance_baseline is None:
            self.performance_baseline = current_metrics
            return

        # Check if performance has degraded
        latency_regression = (current_metrics['avg_latency'] /
                            self.performance_baseline['avg_latency']) > 1.1

        throughput_regression = (current_metrics['throughput'] /
                               self.performance_baseline['throughput']) < 0.9

        if latency_regression or throughput_regression:
            print("Performance regression detected, triggering re-optimization...")

            # Analyze what changed
            workload_drift = self._detect_workload_drift(current_metrics)

            if workload_drift:
                # Re-optimize for new workload pattern
                new_optimized_session = self.adaptive_optimization(workload_drift)
                return new_optimized_session

        return None

    def _detect_workload_drift(self, current_metrics):
        """Detect changes in workload characteristics"""
        if not self.optimization_history:
            return None

        recent_history = self.optimization_history[-10:]  # Last 10 optimization cycles

        # Compare batch size distribution
        historical_batch_sizes = [h['batch_sizes'] for h in recent_history]
        current_batch_sizes = current_metrics['batch_size_distribution']

        # Statistical test for distribution change
        from scipy import stats
        statistic, p_value = stats.ks_2samp(
            np.concatenate(historical_batch_sizes),
            current_batch_sizes
        )

        if p_value < 0.05:  # Significant change detected
            return {
                'batch_sizes': current_batch_sizes,
                'input_shapes': current_metrics['input_shape_distribution'],
                'latency_sla': np.percentile(current_metrics['latency_history'], 95)
            }

        return None

class PerformanceMonitor:
    def __init__(self):
        self.metrics_history = []

    def record_inference(self, latency, batch_size, input_shape):
        """Record inference metrics"""
        self.metrics_history.append({
            'timestamp': time.time(),
            'latency': latency,
            'batch_size': batch_size,
            'input_shape': input_shape
        })

        # Keep only recent history
        if len(self.metrics_history) > 10000:
            self.metrics_history = self.metrics_history[-5000:]

    def get_metrics(self):
        """Get current performance metrics"""
        if not self.metrics_history:
            return {}

        recent_metrics = self.metrics_history[-1000:]  # Last 1000 inferences

        latencies = [m['latency'] for m in recent_metrics]
        batch_sizes = [m['batch_size'] for m in recent_metrics]

        return {
            'avg_latency': np.mean(latencies),
            'p95_latency': np.percentile(latencies, 95),
            'throughput': len(recent_metrics) / (recent_metrics[-1]['timestamp'] - recent_metrics[0]['timestamp']),
            'batch_size_distribution': batch_sizes,
            'input_shape_distribution': [m['input_shape'] for m in recent_metrics],
            'latency_history': latencies
        }

# Example usage
def run_adaptive_optimization_example():
    """Demonstrate adaptive optimization in production"""

    # Initialize optimizer and monitor
    optimizer = RuntimeModelOptimizer("production_model.onnx")
    monitor = PerformanceMonitor()

    # Simulate production workload
    current_session = ort.InferenceSession("production_model.onnx")

    for i in range(1000):  # Simulate 1000 inference requests
        # Generate realistic workload
        batch_size = np.random.choice([1, 4, 8, 16], p=[0.4, 0.3, 0.2, 0.1])
        input_shape = (batch_size, 3, 224, 224)
        input_data = np.random.randn(*input_shape).astype(np.float32)

        # Run inference and measure performance
        start_time = time.time()
        outputs = current_session.run(None, {"input": input_data})
        latency = time.time() - start_time

        # Record metrics
        monitor.record_inference(latency, batch_size, input_shape)

        # Trigger optimization every 100 requests
        if i % 100 == 0 and i > 0:
            new_session = optimizer.continuous_optimization(monitor)
            if new_session:
                current_session = new_session
                print(f"Model re-optimized at request {i}")

    print("Adaptive optimization completed")
    final_metrics = monitor.get_metrics()
    print(f"Final performance: {final_metrics}")
```

This comprehensive enhancement adds detailed, production-ready examples that demonstrate:

---

## Conclusion

This comprehensive guide has covered ONNX Runtime from basic concepts to advanced topics. Here's a summary of what we've learned:

### Key Takeaways:

1. **ONNX Runtime is a powerful, cross-platform ML inference engine** that supports multiple hardware backends and programming languages.

2. **The modular architecture** allows for flexibility in deployment while maintaining high performance through execution providers.

3. **Multiple language bindings** make it accessible to developers across different ecosystems (Python, C++, C#, Java, JavaScript).

4. **Execution providers** are the key to performance, allowing optimal hardware utilization from CPUs to specialized accelerators.

5. **Building and customizing** ONNX Runtime enables optimization for specific use cases and hardware configurations.

6. **Performance optimization** requires understanding of profiling, memory management, and provider-specific tuning.

7. **Contributing to the project** follows established open-source practices with comprehensive testing and documentation requirements.

8. **Advanced features** like custom operators, graph transformers, and training support enable sophisticated ML deployments.

### Next Steps:

1. **Start with simple examples** using pre-built packages
2. **Experiment with different execution providers** to understand performance characteristics
3. **Build from source** to understand the system better
4. **Profile your models** to identify optimization opportunities
5. **Contribute back** to the community with bug fixes or new features
6. **Stay updated** with the latest releases and roadmap

### Resources for Continued Learning:

- **Official Documentation**: [onnxruntime.ai/docs](https://onnxruntime.ai/docs)
- **GitHub Repository**: [github.com/microsoft/onnxruntime](https://github.com/microsoft/onnxruntime)
- **Community Discussions**: [GitHub Discussions](https://github.com/microsoft/onnxruntime/discussions)
- **YouTube Tutorials**: [youtube.com/@ONNXRuntime](https://www.youtube.com/@ONNXRuntime)
- **Sample Code**: [microsoft/onnxruntime-inference-examples](https://github.com/microsoft/onnxruntime-inference-examples)

ONNX Runtime continues to evolve rapidly, with new features, optimizations, and hardware support being added regularly. This guide provides a solid foundation for understanding and working with ONNX Runtime, but the best way to learn is through hands-on experimentation and real-world projects.

Whether you're deploying models in production, optimizing inference performance, or contributing to the open-source project, ONNX Runtime provides the tools and flexibility needed for modern machine learning deployments.
