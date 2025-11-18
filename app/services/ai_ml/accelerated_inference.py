"""
AI/ML Accelerated Inference Engine
Optimized for 130 TOPS compute (NVIDIA GPU + Intel NPU)
Classification: UNCLASSIFIED

Hardware Support:
- NVIDIA RTX 4090 (82 TOPS)
- Intel Core Ultra NPU (48 TOPS)
- CUDA Tensor Cores
- OpenVINO optimization
"""

import torch
import numpy as np
import structlog
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncio
from pathlib import Path

logger = structlog.get_logger()


class AcceleratorType(Enum):
    """Hardware accelerator types"""
    CUDA_GPU = "cuda"
    INTEL_NPU = "npu"
    CPU_AVX512 = "cpu_avx512"
    AUTO = "auto"


@dataclass
class InferenceResult:
    """AI/ML inference result"""
    predictions: np.ndarray
    confidence: float
    latency_ms: float
    device_used: str
    model_name: str


class HardwareDetector:
    """Detect and optimize for available AI hardware"""

    @staticmethod
    def detect_cuda() -> bool:
        """Detect NVIDIA CUDA availability"""
        try:
            return torch.cuda.is_available()
        except:
            return False

    @staticmethod
    def detect_npu() -> bool:
        """Detect Intel NPU availability"""
        try:
            import openvino as ov
            core = ov.Core()
            devices = core.available_devices
            return 'NPU' in devices
        except:
            return False

    @staticmethod
    def get_cuda_device_name() -> Optional[str]:
        """Get CUDA device name"""
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        return None

    @staticmethod
    def get_cuda_compute_capability() -> Optional[Tuple[int, int]]:
        """Get CUDA compute capability"""
        if torch.cuda.is_available():
            return torch.cuda.get_device_capability(0)
        return None

    @staticmethod
    def get_tensor_cores_available() -> bool:
        """Check if Tensor Cores are available"""
        if not torch.cuda.is_available():
            return False

        # Tensor Cores available on compute capability >= 7.0
        cap = torch.cuda.get_device_capability(0)
        return cap[0] >= 7

    @staticmethod
    def optimize_for_hardware():
        """Optimize PyTorch for available hardware"""
        if torch.cuda.is_available():
            # Enable TF32 for Ampere+ GPUs (3x faster)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            # Enable cuDNN autotuner
            torch.backends.cudnn.benchmark = True

            # Enable flash attention if available
            try:
                torch.backends.cuda.enable_flash_sdp(True)
            except:
                pass

            logger.info("CUDA optimizations enabled",
                       device=torch.cuda.get_device_name(0),
                       tensor_cores=HardwareDetector.get_tensor_cores_available())


class AcceleratedInferenceEngine:
    """High-performance inference engine optimized for 130 TOPS"""

    def __init__(self, device: AcceleratorType = AcceleratorType.AUTO):
        self.device = self._select_device(device)
        self.models: Dict[str, torch.nn.Module] = {}

        # Performance optimization
        HardwareDetector.optimize_for_hardware()

        # Mixed precision training/inference
        self.use_fp16 = torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler() if self.use_fp16 else None

        logger.info("Inference engine initialized",
                   device=self.device,
                   fp16=self.use_fp16)

    def _select_device(self, requested: AcceleratorType) -> str:
        """Select best available device"""
        if requested == AcceleratorType.AUTO:
            if HardwareDetector.detect_cuda():
                logger.info("Auto-selected CUDA GPU")
                return "cuda"
            elif HardwareDetector.detect_npu():
                logger.info("Auto-selected Intel NPU")
                return "npu"
            else:
                logger.info("Auto-selected CPU with AVX-512")
                return "cpu"
        else:
            return requested.value

    def load_model(self, model_name: str, model_path: Path,
                  optimize: bool = True) -> bool:
        """Load ML model with optimization"""
        try:
            # Load model
            model = torch.jit.load(model_path)

            # Move to device
            if self.device == "cuda":
                model = model.cuda()

                if optimize:
                    # Optimize for inference
                    model = torch.jit.optimize_for_inference(model)

                    # Compile with TorchScript
                    if hasattr(torch, 'compile'):
                        model = torch.compile(model, mode='max-autotune')

            # Set to eval mode
            model.eval()

            # Store model
            self.models[model_name] = model

            logger.info("Model loaded", name=model_name, device=self.device)
            return True

        except Exception as e:
            logger.error("Model load failed", name=model_name, error=str(e))
            return False

    @torch.no_grad()
    async def infer(self, model_name: str, input_data: np.ndarray) -> InferenceResult:
        """Run accelerated inference"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")

        model = self.models[model_name]

        start_time = asyncio.get_event_loop().time()

        # Convert to tensor
        input_tensor = torch.from_numpy(input_data)

        # Move to device
        if self.device == "cuda":
            input_tensor = input_tensor.cuda()

        # Mixed precision inference
        with torch.cuda.amp.autocast(enabled=self.use_fp16):
            output = model(input_tensor)

        # Move back to CPU and convert to numpy
        predictions = output.cpu().numpy()

        # Calculate confidence
        confidence = float(np.max(torch.softmax(torch.from_numpy(predictions), dim=-1).numpy()))

        latency_ms = (asyncio.get_event_loop().time() - start_time) * 1000

        return InferenceResult(
            predictions=predictions,
            confidence=confidence,
            latency_ms=latency_ms,
            device_used=self.device,
            model_name=model_name
        )

    async def batch_infer(self, model_name: str, input_batch: List[np.ndarray],
                         batch_size: int = 32) -> List[InferenceResult]:
        """Batched inference for better GPU utilization"""
        results = []

        for i in range(0, len(input_batch), batch_size):
            batch = input_batch[i:i+batch_size]

            # Stack into single batch
            batch_array = np.stack(batch)

            # Infer
            result = await self.infer(model_name, batch_array)
            results.append(result)

        return results


class VulnerabilityClassifier:
    """AI-powered vulnerability classification and prioritization"""

    def __init__(self, engine: AcceleratedInferenceEngine):
        self.engine = engine
        self.model_name = "vuln_classifier"

    async def classify_vulnerability(self, vuln_data: Dict) -> Dict:
        """Classify and prioritize vulnerability using AI"""
        # Extract features
        features = self._extract_features(vuln_data)

        # Run inference
        result = await self.engine.infer(self.model_name, features)

        # Interpret results
        severity = self._interpret_severity(result.predictions)
        exploitability = self._calculate_exploitability(result.predictions)
        priority_score = self._calculate_priority(severity, exploitability, vuln_data)

        return {
            'severity': severity,
            'exploitability': exploitability,
            'priority_score': priority_score,
            'confidence': result.confidence,
            'ai_inference_latency_ms': result.latency_ms,
            'recommendations': self._generate_recommendations(severity, exploitability)
        }

    def _extract_features(self, vuln_data: Dict) -> np.ndarray:
        """Extract ML features from vulnerability data"""
        features = []

        # CVSS features
        features.append(vuln_data.get('cvss_base_score', 0.0))
        features.append(vuln_data.get('cvss_temporal_score', 0.0))
        features.append(vuln_data.get('cvss_environmental_score', 0.0))

        # Attack vector (one-hot encoding)
        vectors = ['network', 'adjacent', 'local', 'physical']
        vector = vuln_data.get('attack_vector', 'network')
        features.extend([1.0 if v == vector else 0.0 for v in vectors])

        # Privileges required
        privs = ['none', 'low', 'high']
        priv = vuln_data.get('privileges_required', 'none')
        features.extend([1.0 if p == priv else 0.0 for p in privs])

        # User interaction
        features.append(1.0 if vuln_data.get('user_interaction') == 'none' else 0.0)

        # Impact scores
        features.append(vuln_data.get('confidentiality_impact', 0.0))
        features.append(vuln_data.get('integrity_impact', 0.0))
        features.append(vuln_data.get('availability_impact', 0.0))

        # Exploit availability
        features.append(1.0 if vuln_data.get('exploit_available') else 0.0)

        # Public disclosure
        features.append(1.0 if vuln_data.get('publicly_disclosed') else 0.0)

        # Patch availability
        features.append(1.0 if vuln_data.get('patch_available') else 0.0)

        return np.array(features, dtype=np.float32).reshape(1, -1)

    def _interpret_severity(self, predictions: np.ndarray) -> str:
        """Interpret AI predictions to severity level"""
        severity_idx = np.argmax(predictions[0])
        severities = ['low', 'medium', 'high', 'critical']
        return severities[severity_idx] if severity_idx < len(severities) else 'unknown'

    def _calculate_exploitability(self, predictions: np.ndarray) -> float:
        """Calculate exploitability score from predictions"""
        # Use second output head if available
        if predictions.shape[1] > 4:
            return float(predictions[0][4])
        return 0.5

    def _calculate_priority(self, severity: str, exploitability: float,
                           vuln_data: Dict) -> int:
        """Calculate priority score (0-100)"""
        base_scores = {
            'critical': 90,
            'high': 70,
            'medium': 50,
            'low': 30
        }

        score = base_scores.get(severity, 50)

        # Boost if exploit available
        if vuln_data.get('exploit_available'):
            score += 10

        # Boost if no patch
        if not vuln_data.get('patch_available'):
            score += 10

        # Boost by exploitability
        score += int(exploitability * 10)

        return min(100, score)

    def _generate_recommendations(self, severity: str, exploitability: float) -> List[str]:
        """Generate AI-powered recommendations"""
        recommendations = []

        if severity in ['critical', 'high']:
            recommendations.append("Immediate patching required")
            recommendations.append("Consider isolating affected systems")

        if exploitability > 0.7:
            recommendations.append("High exploitability detected - prioritize remediation")
            recommendations.append("Implement compensating controls immediately")

        if severity == 'critical' and exploitability > 0.8:
            recommendations.append("CRITICAL: Active exploitation likely - emergency response")

        return recommendations


class AnomalyDetector:
    """AI-powered anomaly detection for network traffic and behavior"""

    def __init__(self, engine: AcceleratedInferenceEngine):
        self.engine = engine
        self.model_name = "anomaly_detector"
        self.baseline: Optional[np.ndarray] = None

    async def detect_anomalies(self, network_traffic: np.ndarray) -> Dict:
        """Detect anomalies in network traffic patterns"""
        # Run inference
        result = await self.engine.infer(self.model_name, network_traffic)

        # Calculate anomaly score
        anomaly_score = float(result.predictions[0])

        is_anomalous = anomaly_score > 0.7

        return {
            'is_anomalous': is_anomalous,
            'anomaly_score': anomaly_score,
            'confidence': result.confidence,
            'threat_indicators': self._extract_threat_indicators(result.predictions) if is_anomalous else [],
            'recommended_actions': self._recommend_actions(anomaly_score)
        }

    def _extract_threat_indicators(self, predictions: np.ndarray) -> List[str]:
        """Extract specific threat indicators from predictions"""
        indicators = []

        # Parse prediction vector for known threat patterns
        if predictions.shape[1] > 10:
            if predictions[0][1] > 0.8:  # Port scanning
                indicators.append("Port scanning detected")
            if predictions[0][2] > 0.8:  # DDoS
                indicators.append("Potential DDoS activity")
            if predictions[0][3] > 0.8:  # Data exfiltration
                indicators.append("Unusual data transfer volume")
            if predictions[0][4] > 0.8:  # Lateral movement
                indicators.append("Lateral movement detected")

        return indicators

    def _recommend_actions(self, anomaly_score: float) -> List[str]:
        """Recommend response actions based on anomaly score"""
        actions = []

        if anomaly_score > 0.9:
            actions.append("URGENT: Block source IP immediately")
            actions.append("Initiate incident response")
            actions.append("Preserve logs and network captures")

        elif anomaly_score > 0.7:
            actions.append("Increase monitoring on source")
            actions.append("Review recent activity")
            actions.append("Prepare containment procedures")

        elif anomaly_score > 0.5:
            actions.append("Flag for manual review")
            actions.append("Continue monitoring")

        return actions


# ============================================================================
# 130 TOPS Optimization Utilities
# ============================================================================

class PerformanceOptimizer:
    """Optimize for 130 TOPS hardware configuration"""

    @staticmethod
    def enable_all_optimizations():
        """Enable all available performance optimizations"""
        logger.info("Enabling 130 TOPS optimizations...")

        # CUDA optimizations
        if torch.cuda.is_available():
            # TF32 (3x speedup on Ampere+)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            # cuDNN benchmark
            torch.backends.cudnn.benchmark = True

            # Set optimal memory allocation
            torch.cuda.set_per_process_memory_fraction(0.9)

            # Enable cudnn autotuner
            torch.backends.cudnn.enabled = True

            logger.info("CUDA optimizations enabled",
                       device=torch.cuda.get_device_name(0),
                       memory_gb=torch.cuda.get_device_properties(0).total_memory / 1e9)

        # CPU optimizations
        torch.set_num_threads(torch.get_num_threads())
        torch.set_num_interop_threads(4)

        logger.info("PyTorch optimizations complete")

    @staticmethod
    def benchmark_throughput(model: torch.nn.Module, input_shape: Tuple,
                           iterations: int = 100) -> Dict:
        """Benchmark model throughput"""
        model.eval()

        device = next(model.parameters()).device
        input_tensor = torch.randn(*input_shape).to(device)

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_tensor)

        # Synchronize GPU
        if device.type == 'cuda':
            torch.cuda.synchronize()

        # Benchmark
        start = time.time()
        with torch.no_grad():
            for _ in range(iterations):
                _ = model(input_tensor)

        if device.type == 'cuda':
            torch.cuda.synchronize()

        elapsed = time.time() - start

        throughput = iterations / elapsed
        latency_ms = (elapsed / iterations) * 1000

        return {
            'throughput_inferences_per_sec': throughput,
            'latency_ms': latency_ms,
            'device': str(device),
            'iterations': iterations
        }
