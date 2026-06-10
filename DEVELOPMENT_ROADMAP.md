# VSLAM Development Roadmap - Implementation Tasks

## Execution Plan (Prioritized Tasks)

### **STAGE 1: Foundation (Weeks 1-2)** — Enable Loop Closure Pipeline

#### **1.1 Implement Keyframe Selector** [100 LOC]
**File**: `lidar_mapping/fusion/keyframe_selector.py` (NEW)
**Purpose**: Extract sparse keyframes from dense VO stream

```python
class KeyframeSelector:
    """Select keyframes based on motion/visual quality threshold."""
    
    def __init__(self, min_motion_threshold: float = 0.1):
        self.last_keyframe_pose = None
        self.motion_threshold = min_motion_threshold
        self.keyframes = []
        
    def should_be_keyframe(self, pose: np.ndarray, img_quality: float = None) -> bool:
        """Decide if current pose warrants a new keyframe."""
        if self.last_keyframe_pose is None:
            return True
        # Euclidean distance in translation
        delta = np.linalg.norm(pose[:3, 3] - self.last_keyframe_pose[:3, 3])
        return delta >= self.motion_threshold
    
    def add_keyframe(self, kf_id: int, pose: np.ndarray, descriptors: np.ndarray):
        """Register new keyframe."""
        self.last_keyframe_pose = pose
        self.keyframes.append({
            'id': kf_id,
            'pose': pose.copy(),
            'descriptors': descriptors.copy(),
        })
```

**Integration Point**: 
- Modify `VisualFrontend._tick()` to call `keyframe_selector.should_be_keyframe()`
- Store keyframe descriptors for loop matching

**Tests**:
- `tests/test_keyframe_selector.py`: Motion threshold validation

---

#### **1.2 Add Loop Closure Matcher** [150 LOC]
**File**: `lidar_mapping/fusion/loop_closure.py` (NEW)
**Purpose**: Detect when camera revisits previously seen region

```python
class LoopClosureDetector:
    """Match current frame against all keyframe descriptors."""
    
    def __init__(self, descriptor_match_ratio: float = 0.75):
        self.keyframe_db = []  # List of (kf_id, descriptors)
        self.match_ratio = descriptor_match_ratio
        
    def find_loop_candidates(self, descriptors: np.ndarray, 
                              min_matches: int = 20) -> List[tuple[int, int]]:
        """
        Return list of (keyframe_id, match_count) candidates.
        """
        results = []
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        for kf_id, kf_desc in self.keyframe_db:
            knn = matcher.knnMatch(kf_desc, descriptors, k=2)
            good = 0
            for pair in knn:
                if len(pair) == 2:
                    m, n = pair
                    if m.distance < self.match_ratio * n.distance:
                        good += 1
            if good >= min_matches:
                results.append((kf_id, good))
        
        return sorted(results, key=lambda x: -x[1])  # Sort by count
    
    def verify_loop_with_geometry(self, pts_current: np.ndarray, 
                                   pts_previous: np.ndarray, 
                                   K: np.ndarray) -> Optional[np.ndarray]:
        """
        Verify loop hypothesis with Essential Matrix + RANSAC.
        Returns relative transform (4x4) if loop is valid, else None.
        """
        E, mask = cv2.findEssentialMat(pts_current, pts_previous, K, 
                                        method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None or np.sum(mask) < 8:
            return None
        
        _, R, t, mask = cv2.recoverPose(E, pts_current, pts_previous, K, mask=mask)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.ravel()
        return T
```

**Integration Point**:
- Call in `VisualFrontend._tick()` after feature matching
- Publish loop closure constraints to pose graph

**Tests**:
- `tests/test_loop_closure.py`: Descriptor matching, Essential matrix

---

#### **1.3 Extend FusionState for Loop Constraints** [50 LOC]
**File**: `lidar_mapping/observability/state.py` (MODIFY)
**Purpose**: Track loop closure candidates and constraints

```python
@dataclass
class LoopConstraint:
    """A detected loop closure between two poses."""
    keyframe_id_a: int
    keyframe_id_b: int
    transform: np.ndarray  # 4x4 relative pose
    match_count: int
    timestamp: float = field(default_factory=time.time)

# Add to FusionState:
loop_constraints: List[LoopConstraint] = field(default_factory=list)
loop_constraint_count: int = 0
```

---

### **STAGE 2: Pose Graph Backend (Weeks 2-3)** — Unified Optimization

#### **2.1 Implement Custom 4-DOF Pose Graph Optimizer** [300 LOC]
**File**: `lidar_mapping/fusion/pose_graph_backend.py` (NEW)
**Purpose**: Lightweight pose-graph SLAM without Open3D dependency

```python
class PoseGraphOptimizer:
    """
    Factor-based pose graph for 4-DOF odometry (x, y, z, yaw).
    Assumes gravity-aligned with IMU providing roll/pitch.
    """
    
    class Pose:
        """4-DOF pose: (x, y, z, yaw_rad)"""
        def __init__(self, x: float = 0, y: float = 0, z: float = 0, yaw: float = 0):
            self.pose = np.array([x, y, z, yaw], dtype=np.float64)
    
    class Factor:
        """Measurement between two poses."""
        def __init__(self, pose_id_a: int, pose_id_b: int, 
                     measurement: np.ndarray, information: np.ndarray):
            self.pose_id_a = pose_id_a
            self.pose_id_b = pose_id_b
            self.measurement = measurement  # 4D delta
            self.information = information  # 4x4 covariance
    
    def __init__(self):
        self.poses = []  # List[Pose]
        self.factors = []  # List[Factor]
        self.pose_map = {}  # pose_id → index
    
    def add_pose(self, pose_id: int, pose: np.ndarray):
        """Add a new pose node."""
        if pose_id not in self.pose_map:
            self.pose_map[pose_id] = len(self.poses)
            self.poses.append(PoseGraphOptimizer.Pose(*pose))
    
    def add_odometry_factor(self, pose_id_a: int, pose_id_b: int, 
                           delta: np.ndarray, covariance: np.ndarray):
        """Add odometry measurement (VO, LiDAR, or IMU)."""
        info = np.linalg.inv(covariance) if np.linalg.matrix_rank(covariance) == 4 else np.eye(4)
        self.factors.append(PoseGraphOptimizer.Factor(pose_id_a, pose_id_b, delta, info))
    
    def add_loop_closure_factor(self, pose_id_a: int, pose_id_b: int,
                               transform: np.ndarray, confidence: float = 1.0):
        """Add loop closure constraint (higher confidence than odometry)."""
        delta = transform_to_4dof(transform)
        # Higher confidence = higher information weight
        cov = 0.1 * np.eye(4) * (1.0 / max(confidence, 0.1))
        self.factors.append(PoseGraphOptimizer.Factor(pose_id_a, pose_id_b, delta, cov))
    
    def optimize(self, iterations: int = 5):
        """Run Gauss-Newton optimization (simple but effective)."""
        for iteration in range(iterations):
            H = np.zeros((4 * len(self.poses), 4 * len(self.poses)))
            b = np.zeros(4 * len(self.poses))
            
            for factor in self.factors:
                i_a = self.pose_map[factor.pose_id_a]
                i_b = self.pose_map[factor.pose_id_b]
                
                # Residual: predicted - measured
                delta_pred = self.poses[i_b].pose - self.poses[i_a].pose
                residual = delta_pred - factor.measurement
                
                # Jacobian for 4-DOF (simplified linear)
                J_a = -np.eye(4)
                J_b = np.eye(4)
                
                # Accumulate
                H[i_a*4:(i_a+1)*4, i_a*4:(i_a+1)*4] += J_a.T @ factor.information @ J_a
                H[i_b*4:(i_b+1)*4, i_b*4:(i_b+1)*4] += J_b.T @ factor.information @ J_b
                H[i_a*4:(i_a+1)*4, i_b*4:(i_b+1)*4] += J_a.T @ factor.information @ J_b
                
                b[i_a*4:(i_a+1)*4] += J_a.T @ factor.information @ residual
                b[i_b*4:(i_b+1)*4] += J_b.T @ factor.information @ residual
            
            # Fix first pose (gauge freedom)
            H[:4, :4] = np.eye(4)
            b[:4] = 0
            
            # Solve and update
            delta_x = np.linalg.solve(H, b)
            for i, pose in enumerate(self.poses):
                pose.pose += 0.1 * delta_x[i*4:(i+1)*4]  # Damped update
```

**Integration Point**:
- Initialize in `run_stationary.py` or `sensor_hub.py`
- Add factors from VO, LiDAR, and loop closure detections
- Call optimize on loop closure or at fixed interval

**Tests**:
- `tests/test_pose_graph.py`: Linear solve, pose convergence

---

#### **2.2 Integrate ICP Scan Matching (Conditional)** [50 LOC]
**File**: `lidar_mapping/fusion/lidar_frontend.py` (MODIFY)
**Purpose**: Switch from `imu_only` mode to `icp` mode with pose graph

```python
# Modify LidarFrontend.__init__ to accept pose_graph_backend:
def __init__(self, ..., pose_graph_backend: Optional[PoseGraphOptimizer] = None):
    self.pose_graph = pose_graph_backend
    # ... existing code

# In _loop, enable ICP when pose_graph is provided:
if self.pose_source == "icp" and self.pose_graph is not None:
    result = self.mapper.add_scan(pts, transform_hint=imu_rotation_hint)
    if result.converged:
        # Add odometry factor to pose graph
        self.pose_graph.add_odometry_factor(
            prev_pose_id, curr_pose_id, 
            result.transform, cov=np.eye(4) * 0.01
        )
```

---

### **STAGE 3: Loop Closure Integration (Week 3)** — Connect Detector to Optimizer

#### **3.1 Add Loop Closure to Main Fusion Loop** [100 LOC]
**File**: `apps/run_stationary.py` (MODIFY)
**Purpose**: Integrate loop closure detection into real-time pipeline

```python
# In main():
loop_detector = LoopClosureDetector(descriptor_match_ratio=0.75)
pose_graph = PoseGraphOptimizer()

# In cam_state_mux_loop (where overlay is processed):
    candidates = loop_detector.find_loop_candidates(descriptors)
    for kf_id, match_count in candidates[:3]:  # Top 3
        if match_count > 30:
            # Geometric verification
            T_loop = loop_detector.verify_loop_with_geometry(
                current_keypoints, keyframe_points, K
            )
            if T_loop is not None:
                # Add loop constraint
                pose_graph.add_loop_closure_factor(kf_id, current_kf_id, T_loop)
                print(f"Loop closure detected: KF{kf_id} → KF{current_kf_id} ({match_count} matches)")
                
                # Optimize
                pose_graph.optimize(iterations=3)
```

---

### **STAGE 4: Testing & Validation (Week 4)**

#### **4.1 Unit Tests**
**File**: `tests/test_loop_closure_integration.py` (NEW)
```python
def test_loop_closure_same_frame():
    """Verify loop match with identical keyframes."""
    detector = LoopClosureDetector()
    descriptors = np.random.randint(0, 256, (100, 32), dtype=np.uint8)
    detector.keyframe_db.append((0, descriptors))
    
    candidates = detector.find_loop_candidates(descriptors)
    assert candidates[0][0] == 0  # Should match itself
    assert candidates[0][1] > 50  # Many matches
```

#### **4.2 Integration Test**
**File**: `tests/test_end_to_end.py` (NEW)
```python
def test_multi_loop_trajectory():
    """Test multi-loop trajectory with drift correction."""
    # Run 3 loops of stationary rotation test
    # Verify drift < 5% per loop with loop closures
    # Compare against no loop closure (baseline)
```

#### **4.3 Benchmark Script**
**File**: `apps/benchmark_vslam.py` (NEW)
```python
def main():
    """
    Run VSLAM on recorded trajectory and measure:
    - Relative Pose Error (RPE)
    - Absolute Trajectory Error (ATE)
    - Loop closure precision/recall
    """
```

---

## Timeline & Milestones

| Week | Task | Output | Validation |
|------|------|--------|-----------|
| 1 | Keyframes + Loop Detector | `keyframe_selector.py`, `loop_closure.py` | Unit tests pass |
| 2 | Pose Graph Optimizer | `pose_graph_backend.py` | Optimization converges |
| 3 | Main loop integration | Modified `run_stationary.py` | Loop closures detected in live test |
| 4 | Testing & benchmarking | `benchmark_vslam.py`, results | Drift < 2% per loop |

---

## File Modification Checklist

### New Files to Create
- [ ] `lidar_mapping/fusion/keyframe_selector.py`
- [ ] `lidar_mapping/fusion/loop_closure.py`
- [ ] `lidar_mapping/fusion/pose_graph_backend.py`
- [ ] `apps/benchmark_vslam.py`
- [ ] `tests/test_keyframe_selector.py`
- [ ] `tests/test_loop_closure.py`
- [ ] `tests/test_pose_graph.py`
- [ ] `tests/test_loop_closure_integration.py`

### Files to Modify
- [ ] `lidar_mapping/observability/state.py` (add LoopConstraint dataclass)
- [ ] `lidar_mapping/fusion/sensor_hub.py` (optional: add loop detector thread)
- [ ] `lidar_mapping/fusion/visual_frontend.py` (call keyframe_selector)
- [ ] `lidar_mapping/fusion/lidar_frontend.py` (enable ICP mode)
- [ ] `apps/run_stationary.py` (main loop integration)
- [ ] `pyproject.toml` (no new deps needed)

---

## Alternative: Lite ICP for ARM

**If ICP registration fails on Orange Pi 5**:
Implement simple point-to-plane ICP in Python (~200 LOC):

```python
def icp_point_to_plane(source: np.ndarray, target: np.ndarray, 
                       max_iter: int = 20, threshold: float = 0.01):
    """Lightweight ICP using point-to-plane residuals."""
    # 1. Estimate target normals (PCA on local neighborhood)
    # 2. For each iteration:
    #    a. Find nearest neighbors (KD-tree)
    #    b. Compute point-to-plane distances
    #    c. Solve via SVD (no RANSAC needed)
    # 3. Return transform + fitness score
```

Would be integrated into `lidar_mapping/processing/registration.py`.

---

## Success Criteria for STAGE 1-4

✅ Loop closures detected on multi-loop trajectory  
✅ Pose graph optimization converges in < 100ms  
✅ Drift corrected to < 2% per loop  
✅ All unit tests passing  
✅ Benchmark script produces repeatable results  

**Next Phase (STAGE 5)**: Deploy on real motion trajectories (beyond stationary rotation).
