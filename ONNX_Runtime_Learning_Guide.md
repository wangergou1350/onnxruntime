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

### Graph Optimization Levels

```python
# Set optimization level
options = ort.SessionOptions()
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
# Options: ORT_DISABLE_ALL, ORT_ENABLE_BASIC, ORT_ENABLE_EXTENDED, ORT_ENABLE_ALL
```

### IO Binding for High Performance

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

#### 1. Quantization:
```python
# Post-training quantization
from onnxruntime.quantization import quantize_dynamic

quantize_dynamic("model.onnx", "model_quantized.onnx",
                weight_type=QuantType.QUInt8)
```

#### 2. Model Pruning:
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

### Training Support

ONNX Runtime also supports training workloads:

#### Training Configuration:
```python
import onnxruntime.training as orttraining

# Create training session
training_session = orttraining.TrainingSession(
    training_model_path="training_model.onnx",
    eval_model_path="eval_model.onnx",
    optimizer_model_path="optimizer_model.onnx"
)

# Training loop
for epoch in range(num_epochs):
    for batch in train_dataloader:
        # Forward pass
        loss = training_session.train_step(batch.inputs, batch.labels)

        # Backward pass and optimization handled internally

        if step % eval_interval == 0:
            eval_loss = training_session.eval_step(eval_batch.inputs, eval_batch.labels)
```

### Debugging and Profiling

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
