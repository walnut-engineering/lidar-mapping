# VSLAM Architecture Decisions & Trade-offs

## Quick Decision Matrix

### **Q1: Which backend optimizer to use?**

| Backend | Pros | Cons | Effort | Dependency |
|---------|------|------|--------|-----------|
| **Custom 4-DOF** ✅ | Lightweight, controllable, works on ARM | Limited to 4D, less robust | 300 LOC | None |
| Ceres Solver | Mature, flexible, proven | Harder build, not prebuilt on ARM | 400 LOC | C++ binding |
| GTSAM | Feature-rich, academic standard | Heavy dependency, harder to install | 500 LOC | C++ binding |
| Open3D | Already ported in code | **Not available on aarch64** | Done | ❌ Blocked |

**Recommendation**: **Custom 4-DOF** — Start here, can switch later if needed.

---

### **Q2: Loop closure: Visual or LiDAR?**

| Approach | Pros | Cons | Effort | Robustness |
|----------|------|------|--------|-----------|
| **Visual (ORB)** ✅ | Fast, texture-agnostic, existing ORB | Light variation, repetitive patterns fail | 100 LOC | Medium |
| LiDAR Scan Context | Great on structured scenes | Heavy computation, requires full scans | 150 LOC | High on indoor |
| Hybrid (Visual + LiDAR) | Robust to both dark and featureless | More complex, higher false-positive rate | 200 LOC | Very High |

**Recommendation**: **Visual (ORB descriptor matching)** — Sufficient for initial VSLAM, can add LiDAR later.

---

### **Q3: When to add loop constraints to graph?**

| Strategy | Timing | Pros | Cons |
|----------|--------|------|------|
| **Aggressive** ✅ | Add any match > threshold | Fast loop closure | Higher false positives |
| Conservative | Add after geometric verification + re-observe | Few false positives | Slower loop closure |
| Delayed | Batch add every 100 frames | Efficient optimization | Delayed drift correction |

**Recommendation**: **Aggressive with geometric verification** — Best effort balance.

---

### **Q4: Keyframe selection criteria?**

| Criteria | Threshold | Pros | Cons |
|----------|-----------|------|------|
| **Motion-only** ✅ | 0.05-0.1 m translation | Simple, real-time | May miss important frames |
| Feature quality | Median descriptor entropy | Captures info-rich frames | Slower computation |
| Hybrid | Motion AND quality above threshold | Best coverage | More parameters to tune |

**Recommendation**: **Motion-only (0.05 m threshold)** — Start simple, adjust if needed.

---

## Implementation Priority Justification

### **Why Keyframe Selection First?**
1. **Minimal dependency**: Pure geometry, no optimization needed
2. **Blocking feature**: Loop detection can't work without keyframes
3. **Fast feedback**: Can validate immediately on live data
4. **Decoupled**: Doesn't depend on other modules

### **Why Loop Closure Second?**
1. **Tests pose graph**: Found bugs early with simple graph
2. **Validates backend**: Before complex factors, verify basic operation
3. **Incremental value**: System works better *now* (not after all 4 stages)
4. **Visibility**: Loop closure is obviously when it works

### **Why Pose Graph Third?**
1. **Foundation ready**: Keyframes + loop closure working first
2. **Simpler optimization**: Only 4-DOF, easier to debug
3. **Natural integration**: Loops already provide constraints
4. **Testable**: Can benchmark immediately

---

## Risk Mitigation

### **Risk 1: Pose Graph Divergence**
- **Symptom**: Optimization makes poses worse, not better
- **Root Cause**: Singular matrix, bad initial guess, outlier loops
- **Mitigation**: 
  - Add gauge fixing (first pose frozen)
  - Reject loops with low confidence
  - Log residuals per iteration
- **Fallback**: Use weighted averaging instead of optimization (Phase 2)

### **Risk 2: Loop Closure False Positives**
- **Symptom**: Incorrect loop closures warp trajectory
- **Root Cause**: Repetitive textures, changing lighting
- **Mitigation**:
  - Require Essential Matrix verification
  - Spatially separate keyframes (don't match adjacent)
  - Track false positive rate via validation
- **Fallback**: Disable loop closure, use pure odometry

### **Risk 3: Performance Regression on ARM**
- **Symptom**: Optimization takes > 100ms per frame, stalls pipeline
- **Root Cause**: Matrix inversions, lack of vectorization
- **Mitigation**:
  - Profile before optimizing
  - Use sparse matrices (scipy.sparse)
  - Batch optimize every 10 frames, not every frame
- **Fallback**: Optimize offline after recording

### **Risk 4: Open3D Not Available on aarch64**
- **Current Status**: ✅ Already worked around (custom 4-DOF)
- **Mitigation**: Don't depend on Open3D for core loop closure
- **Fallback**: Keep lite ICP in `registration.py` as backup

---

## Success Indicators (Per Stage)

### **STAGE 1 Success**: Keyframes + Loop Detector
✅ Can identify revisited frames from live camera  
✅ False positive rate < 10% on known multi-loop sequence  
✅ Descriptors stored and retrieved reliably  

**Validation**: Record 2-loop trajectory, verify correct loop matches

---

### **STAGE 2 Success**: Pose Graph Backend
✅ Optimizer converges on synthetic trajectory  
✅ Error norm decreasing over iterations  
✅ Runs in < 50ms per optimize() call  

**Validation**: Compare pose graph output vs. initial poses on test data

---

### **STAGE 3 Success**: Full Integration
✅ Live loop closures detected during rotation test  
✅ Graph corrections applied to trajectory  
✅ Snapshots show corrected camera overlay  

**Validation**: Run 3-loop rotation test, verify drift reduction

---

### **STAGE 4 Success**: Validation Suite
✅ Benchmark script produces repeatable measurements  
✅ Loop closure precision > 90%  
✅ Drift corrected to < 2% per loop  

**Validation**: Compare against known ground truth or IMU-only baseline

---

## Trade-off Analysis: MVP vs. Full VSLAM

### **MVP (Minimum Viable SLAM)** — 2 weeks
- ✅ Keyframe selection
- ✅ Loop detection (visual)
- ✅ Simple pose averaging (not full optimization)
- ✅ Multi-loop trajectory with error < 5%

**Blocker**: No global optimization yet

### **Production VSLAM** — 4 weeks
- ✅ Keyframes + keyframe graph
- ✅ Loop closure with geometric verification
- ✅ Factor-based pose graph optimization
- ✅ Multi-loop trajectory with error < 2%
- ✅ Relocalization on loop closure
- ✅ Long-term deployment validation

**Advantage**: Drift-corrected, globally consistent map

---

## Recommended Starting Point

### **Option A: Aggressive** (2-week sprint)
Start with **STAGE 1 + 2** (keyframes + pose graph).  
**Target**: Live loop closure detection + graph optimization.  
**Validation**: Show loop closure working on multi-loop rotation test.

### **Option B: Conservative** (4-week deployment)
All **STAGES 1-4**.  
**Target**: Production-ready VSLAM with benchmarking.  
**Validation**: Complete benchmark suite + ground truth comparison.

**Recommendation**: **Option A (aggressive)** → If successful, do Option B in parallel with field testing.

---

## Code Quality Checkpoints

| Checkpoint | Condition | Validate |
|-----------|-----------|----------|
| Unit tests | All pass before integration | pytest runs without errors |
| Integration test | Live system processes loop closure | Observability snapshot shows loop markers |
| Performance | Optimization < 50ms | Profiler shows allocation counts |
| Robustness | Handles missing data | Incomplete keyframes don't crash |

---

## Next 24 Hours

1. **Review & feedback** on this roadmap
2. **Start STAGE 1.1**: Create `keyframe_selector.py`
3. **Test immediately**: Validate on live rotation test
4. **Iterate**: Adjust motion threshold based on real data

**Expected outcome**: First commit with working keyframe extraction and 10+ loop candidates per multi-loop test.
