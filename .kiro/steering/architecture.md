# Architecture

## Pipeline Overview

Conductor follows this processing pipeline:

```
Camera → Hand Landmark Detection → Feature Extraction → Gesture Recognition → Gesture State Machine → Intent Resolution → Action Dispatcher → Browser Action Adapter
```

## Architectural Principles

### Layer Separation

- The vision layer must **never** directly execute browser or OS actions
- Gesture recognition produces semantic `GestureEvents`
- `GestureEvents` are resolved into semantic browser intents before actions occur
- Only the Action Dispatcher may invoke browser actions

### Gesture Processing

- **Normalization**: Hand measurements should be normalized by hand scale rather than relying on raw pixel distances whenever possible
- **Continuous gestures**: Must have explicit `engage`, `active`, and `release` states
- **Discrete gestures**: Must use debouncing, hysteresis, and re-arming where necessary to prevent repeated actions
- **Context-sensitive gestures**: A gesture may have different semantic meanings only when the active interaction state makes the distinction explicit. In the current MVP, open palm releases navigation mode when navigation is active and closes a tab only while idle.

### Safety & Reliability

- Low-confidence detections, stale frames, camera failures, and tracking loss must **never** produce browser actions
- The system must fail gracefully when hardware is unavailable
- Frame timestamps must be propagated from camera capture through detection, features, and gesture candidates so stale-frame suppression compares values from the same frame.

### Extensibility

- The initial browser adapter may use PyAutoGUI, but the design must allow a future browser-extension adapter without changing the gesture engine

### Decoupling & Testing

- Gesture recognition must be independently testable without a physical camera
- Keep computer-vision, interaction/state-management, browser actions, and UI concerns decoupled
- Prefer small, testable components and explicit interfaces over implicit coupling

### Performance

- Avoid introducing cloud inference into the real-time gesture loop

## Module Boundaries

| Module | Responsibility | Outputs |
|--------|---------------|---------|
| Camera | Frame capture | Raw frames |
| Hand Landmark Detection | MediaPipe processing | Hand landmarks |
| Feature Extraction | Compute geometric features | Normalized features |
| Gesture Recognition | Classify gesture types | Gesture candidates |
| Gesture State Machine | Track gesture lifecycle | GestureEvents |
| Intent Resolution | Map gestures to actions | Browser intents |
| Action Dispatcher | Execute browser actions | Side effects |

## Cross-References

- See `tech.md` for technology stack and dependency constraints
- See `product.md` for supported browser actions and product principles
