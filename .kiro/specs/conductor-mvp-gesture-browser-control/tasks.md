# Implementation Plan: Conductor MVP Gesture Browser Control

## Overview

This implementation plan follows the architectural pipeline from camera capture through gesture recognition to browser action execution. The design uses Python 3.12 with OpenCV, MediaPipe, Tkinter, and PyAutoGUI. All gesture recognition runs locally with no cloud APIs.

## Tasks

- [x] 1. Set up project foundation
  - [x] 1.1 Add core dependencies to pyproject.toml
    - Add opencv-python, mediapipe, pyautogui, pytest, hypothesis dependencies
    - Configure pytest and hypothesis settings
    - _Requirements: 18.1_
  
  - [x] 1.2 Create configuration system in config/
    - Create config/__init__.py
    - Create config/config.py with all configuration dataclasses (CameraConfig, DetectorConfig, GestureConfig, StateMachineConfig, IntentConfig, HUDConfig, Config)
    - Implement load_config function with TOML file loading and defaults
    - Create default config/config.toml with all threshold values
    - _Requirements: 18.1, 18.2, 18.3_
  
  - [x] 1.3 Create core data models in src/models.py
    - Define Frame, FrameMetadata dataclasses
    - Define Landmark, HandLandmarks, DetectionResult dataclasses
    - Define NormalizedFeatures dataclass
    - Define GestureType enum and GestureCandidate, SwipeMetadata dataclasses
    - Define NavModeState, the close-gesture lifecycle state (currently retained as FistState for compatibility), and SwipeState enums
    - Define GestureEvent, StateMachineState dataclasses
    - Define IntentType enum and BrowserIntent dataclass
    - _Requirements: 1.2, 2.2, 3.1, 4.1, 5.1, 6.1, 7.1, 16.1_

- [x] 2. Implement camera module
  - [x] 2.1 Create Camera class in src/camera/camera.py
    - Implement Frame and FrameMetadata capture using OpenCV VideoCapture
    - Implement start(), stop(), read_frame(), get_metadata(), is_available() methods
    - Handle camera unavailable and capture failure errors gracefully
    - _Requirements: 1.1, 1.2, 1.3, 1.4_
  
  - [x] 2.2 Write unit tests for camera module
    - Test frame capture with mocked VideoCapture
    - Test error handling when camera unavailable
    - Test recovery when camera becomes available
    - _Requirements: 1.3, 1.4_

- [x] 3. Implement hand landmark detection
  - [x] 3.1 Create HandLandmarkDetector class in src/vision/detector.py
    - Wrap MediaPipe Hands for local processing
    - Implement process() method that takes Frame and returns DetectionResult
    - Extract landmarks and confidence from MediaPipe results
    - Handle no-hand detection case (return DetectionResult with landmarks=None)
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_
  
  - [x] 3.2 Write unit tests for hand landmark detector
    - Test with mocked MediaPipe results
    - Test no-hand detection output
    - Test low-confidence output
    - _Requirements: 2.3, 2.4, 19.1_

- [x] 4. Implement feature extraction
  - [x] 4.1 Create FeatureExtractor class in src/vision/features.py
    - Implement extract() method that computes NormalizedFeatures from HandLandmarks
    - Compute hand_scale reference (max distance from wrist to fingertips)
    - Compute normalized pinch_distance (thumb tip to middle finger tip)
    - Compute hand_center (centroid of landmarks)
    - Compute finger_extensions using normalized joint geometry (not raw Y comparison)
    - Compute openness_score from finger extensions
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_
  
  - [ ]* 4.2 Write property tests for feature normalization
    - **Property 1: Feature Normalization Bounded Range** - All normalized distance features in [0, 2], position features in [0, 1]
    - **Validates: Requirements 3.1, 3.2**
  
- [x] 4.3 Write property tests for pinch distance
    - **Property 2: Pinch Distance Non-Negative** - Computed pinch_distance always non-negative
    - **Validates: Requirements 3.3**
    - Implemented in `tests/property/test_feature_properties.py`
  
  - [x] 4.4 Write unit tests for feature extractor
    - Test pinch distance calculation with known landmark positions
    - Test finger extension calculations
    - Test hand scale normalization
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

- [x] 5. Implement gesture recognizer - pose stability
  - [x] 5.1 Create GestureRecognizer class skeleton in src/gestures/recognizer.py
    - Define GestureType enum and GestureCandidate dataclass
    - Implement recognize() method interface
    - Create SwipeMetadata dataclass for swipe candidates
    - _Requirements: 4.1, 4.3_
  
  - [x] 5.2 Implement pose classification stability in GestureRecognizer
    - Classify open palm, Victory, pointing, and thumb poses from landmarks
    - Apply classification hysteresis to prevent frame-to-frame flicker
    - Propagate the camera timestamp into each gesture candidate
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_
  
  - [ ]* 5.3 Write property tests for pose stability behavior
    - **Property 3: Victory Stability** - Victory must remain stable before navigation engagement
    - **Property 4: Navigation Stability** - Short alternating pose sequences must not oscillate navigation state
    - **Property 5: Open-Palm Release** - Stable open palm releases active navigation mode
    - **Validates: Requirements 4.1, 4.3, 4.4**
  
  - [x] 5.4 Write unit tests for pose classification
    - Test open-palm, Victory, and thumb pose classification
    - Test hysteresis prevents oscillation
    - Test ambiguous closed-hand poses are ignored
    - _Requirements: 4.1, 4.3, 4.4_

- [x] 6. Implement gesture recognizer - swipe detection
  - [x] 6.1 Implement swipe detection in GestureRecognizer
    - Implement bounded frame buffer for temporal analysis
    - Track hand_center over swipe_window_frames
    - Compute net_displacement, elapsed_time, velocity, direction_consistency
    - Output SWIPE_LEFT or SWIPE_RIGHT when all thresholds met
    - Reject movements below displacement, velocity, or consistency thresholds
    - Reset swipe buffer when navigation mode is engaged
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7_
  
  - [ ]* 6.2 Write property tests for swipe rejection criteria
    - **Property 12: Swipe Displacement Threshold** - Reject if net displacement below threshold
    - **Property 13: Swipe Velocity Threshold** - Reject if velocity below threshold
    - **Property 14: Swipe Direction Consistency** - Reject if direction inconsistent
    - **Validates: Requirements 8.4, 8.5, 8.6**
  
  - [x] 6.3 Write unit tests for swipe detection
    - Test valid swipe detection with known frame sequences
    - Test rejection below displacement threshold
    - Test rejection below velocity threshold
    - Test rejection with inconsistent direction
    - _Requirements: 8.3, 8.4, 8.5, 8.6_

- [x] 7. Implement gesture recognizer - open palm and thumb poses
  - [x] 7.1 Implement open-palm and thumb-direction detection in GestureRecognizer
    - Detect open palm using normalized hand-scale measurements
    - Detect thumb up/down only with sufficient normalized reach and vertical margin
    - Do not emit closed-fist candidates from live camera input
    - _Requirements: 11.1, 11.2, 11.3_
  
  - [x] 7.2 Write unit tests for open-palm and thumb pose detection
    - Test open-palm detection with known extended finger positions
    - Test thumb up/down direction and stable pose handling
    - Test ambiguous closed-hand poses are not emitted
    - _Requirements: 11.1, 11.2, 11.3_

- [x] 8. Implement gesture state machine - navigation lifecycle
  - [x] 8.1 Create GestureStateMachine class in src/interaction/state_machine.py
    - Define NavModeState, FistState (retained as the close lifecycle compatibility state), and SwipeState enums
    - Define GestureEvent, StateMachineState dataclasses
    - Implement process() method interface
    - _Requirements: 5.1, 5.3_
  
  - [x] 8.2 Implement navigation state machine transitions
    - IDLE → ENGAGED on held Victory, emit nav_engage event
    - ENGAGED → ACTIVE on continued Victory
    - ACTIVE → IDLE on held open palm, emit nav_release event
    - Store nav_engage_position for navigation delta calculation
    - _Requirements: 5.1, 5.2, 5.3, 5.4_
  
  - [ ]* 8.3 Write property tests for navigation state transitions
    - **Property 6: Navigation State Machine Transitions** - Correct transitions and event emission
    - **Property 7: No Events During Continuous Navigation** - No events solely because navigation remains active
    - **Validates: Requirements 5.1, 5.2, 5.3**
  
  - [x] 8.4 Write unit tests for navigation state machine
    - Test full navigation lifecycle (idle → engaged → active → released)
    - Test nav_engage_position is stored correctly
    - Test no events during continuous active state
    - _Requirements: 5.1, 5.2, 5.3, 5.4_

- [x] 9. Implement gesture state machine - swipe and open-palm close
  - [x] 9.1 Implement swipe state machine with re-arm
    - NEUTRAL → TRIGGERED on valid swipe candidate, emit Swipe_Left/Swipe_Right event
    - TRIGGERED → WAIT_FOR_REARM after emitting event
    - WAIT_FOR_REARM → NEUTRAL when hand returns to neutral position
    - Reject swipe candidates when Frame_Confidence below threshold
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_
  
  - [x] 9.2 Implement idle open-palm close state machine with hold confirmation
    - NEUTRAL → CANDIDATE on first idle OPEN_PALM candidate
    - CANDIDATE → CONFIRMED after hold_duration, emit Tab_Close event
    - CANDIDATE → NEUTRAL if open palm is released before hold_duration
    - CONFIRMED → WAIT_FOR_REARM after emitting event
    - WAIT_FOR_REARM → NEUTRAL when hand opens
    - No additional events in CONFIRMED or WAIT_FOR_REARM states
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7_
  
  - [ ]* 9.3 Write property tests for swipe state machine
    - **Property 15: Swipe Single Action** - Exactly one event per valid swipe
    - **Property 16: Swipe Re-Arm** - Reject subsequent swipes until hand returns to neutral
    - **Validates: Requirements 9.2, 9.3, 9.4**
  
  - [ ]* 9.4 Write property tests for open-palm close state machine
    - **Property 17: Open-Palm Close Hold Duration** - No Tab_Close if hold < duration
    - **Property 18: Open-Palm Close Confirm Emits Event** - Exactly one Tab_Close after hold duration
    - **Property 19: Open-Palm Close No Repeat Events** - No additional events in CONFIRMED/WAIT_FOR_REARM
    - **Validates: Requirements 12.2, 12.3, 12.6, 12.7**
  
  - [x] 9.5 Write unit tests for swipe and open-palm close state machines
    - Test swipe emits exactly one event
    - Test swipe re-arm requirement
    - Test idle open-palm hold confirmation timing
    - Test open-palm early release (no event)
    - _Requirements: 9.2, 9.3, 9.4, 12.2, 12.3, 12.7_

- [x] 10. Implement safety suppression
  - [x] 10.1 Implement suppression conditions in GestureStateMachine
    - Suppress when Frame_Confidence < confidence_threshold
    - Reset to neutral when no hand detected
    - Suppress for cooldown period after tracking jump detected
    - Suppress when frame older than staleness_threshold
    - Never emit events during suppressed state
    - _Requirements: 14.1, 14.2, 14.3, 14.4, 14.5_
  
  - [ ]* 10.2 Write property tests for safety suppression
    - **Property 20: Low Confidence Suppression** - Suppress all transitions when confidence below threshold
    - **Property 21: No Hand Reset** - Reset to neutral when no hand detected
    - **Property 22: Tracking Jump Suppression** - Suppress for cooldown after large coordinate jump
    - **Property 23: Stale Frame Suppression** - Suppress when frame too old
    - **Property 24: Suppressed State No Events** - No events during suppression
    - **Validates: Requirements 14.1, 14.2, 14.3, 14.4, 14.5**
  
  - [x] 10.3 Write unit tests for safety suppression
    - Test low confidence suppression
    - Test no-hand reset behavior
    - Test tracking jump detection and cooldown
    - Test stale frame rejection
    - _Requirements: 14.1, 14.2, 14.3, 14.4, 14.5_

- [x] 11. Checkpoint - Core pipeline complete
  - Ensure all tests pass, ask the user if questions arise.

- [x] 12. Implement intent resolver
  - [x] 12.1 Create IntentResolver class in src/intent/resolver.py
    - Define IntentType enum and BrowserIntent dataclass
    - Implement resolve() method that maps GestureEvents to BrowserIntents
    - Implement update_nav_active() for incremental scroll generation
    - _Requirements: 6.1, 7.1, 10.1, 10.2, 13.1_
  
  - [x] 12.2 Implement scroll intent generation with incremental movement
    - Track previous_position for incremental delta calculation
    - Calculate incremental vertical movement (delta_y)
    - Apply deadzone - ignore movement within deadzone
    - Emit scroll intent proportional to (|delta_y| - deadzone) × scale_factor
    - Emit SCROLL_UP for upward movement, SCROLL_DOWN for downward
    - Update previous_position after processing (stationary hand = no intents)
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_
  
  - [x] 12.3 Implement tab-switch intent with re-arm
    - Track horizontal displacement from engage_position
    - Emit NEXT_TAB when displacement >= tab_switch_threshold to right
    - Emit PREVIOUS_TAB when displacement <= -tab_switch_threshold to left
    - Require hand to return within re_arm_threshold before next tab-switch
    - Use normalized displacement measurements
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_
  
  - [x] 12.4 Implement navigation intent mapping
    - Map Swipe_Right event to BROWSER_FORWARD intent
    - Map Swipe_Left event to BROWSER_BACK intent
    - Map Tab_Close event to CLOSE_TAB intent
    - _Requirements: 10.1, 10.2, 13.1_
  
  - [ ]* 12.5 Write property tests for scroll intent generation
    - **Property 8: Scroll Intent Requires Movement** - No scroll intents for stationary hand
    - **Property 9: Scroll Intent Proportional to Displacement** - Correct magnitude calculation
    - **Property 10: Scroll Direction Correct** - Correct SCROLL_UP/SCROLL_DOWN for direction
    - **Validates: Requirements 6.2, 6.3, 6.4**
  
  - [ ]* 12.6 Write property tests for tab-switch intent
    - **Property 11: Tab Switch Requires Re-Arm** - No second intent until hand returns within re-arm threshold
    - **Validates: Requirements 7.4**
  
  - [ ]* 12.7 Write property tests for intent mapping
    - **Property 25: Intent Mapping Consistency** - Correct event-to-intent mapping
    - **Validates: Requirements 10.1, 10.2, 13.1**
  
  - [x] 12.8 Write unit tests for intent resolver
    - Test scroll intent generation with known movements
    - Test deadzone behavior
    - Test tab-switch re-arm logic
    - Test navigation intent mapping
    - _Requirements: 6.2, 6.3, 6.4, 7.4, 10.1, 10.2, 13.1_

- [x] 13. Implement browser action adapter
  - [x] 13.1 Create BrowserActionAdapter class in src/actions/adapter.py
    - Implement scroll_up(amount) and scroll_down(amount) using PyAutoGUI
    - Implement next_tab() with platform-specific keyboard shortcut (Ctrl+Tab)
    - Implement previous_tab() with platform-specific shortcut (Ctrl+Shift+Tab)
    - Implement browser_back() with platform-specific shortcut (Alt+Left / Cmd+Left)
    - Implement browser_forward() with platform-specific shortcut (Alt+Right / Cmd+Right)
    - Implement close_tab() with platform-specific shortcut (Ctrl+W / Cmd+W)
    - Detect OS and use appropriate shortcuts
    - _Requirements: 17.1, 17.2, 17.3, 17.4, 17.5, 17.6_
  
  - [x] 13.2 Write unit tests for browser action adapter
    - Test with mocked PyAutoGUI to verify key combinations
    - Test platform-specific shortcut selection
    - Verify browser tab shortcuts used (not OS app-switching shortcuts)
    - _Requirements: 17.1, 17.2, 17.3, 17.4, 17.5, 17.6_

- [x] 14. Implement action dispatcher
  - [x] 14.1 Create ActionDispatcher class in src/actions/dispatcher.py
    - Implement dispatch(intent) method as single entry point
    - Route intents to appropriate BrowserActionAdapter methods
    - Match IntentType to adapter method calls
    - _Requirements: 16.1, 16.2, 16.3, 16.4, 16.5_
  
  - [x] 14.2 Write unit tests for action dispatcher
    - Test dispatch routing for each intent type
    - Verify adapter is called with correct parameters
    - Verify ActionDispatcher is only component that invokes adapter
    - _Requirements: 16.1, 16.2, 16.3, 16.4, 16.5, 19.5_

- [x] 15. Checkpoint - Action execution complete
  - Ensure all tests pass, ask the user if questions arise.

- [x] 16. Implement HUD display
  - [x] 16.1 Create HUD class in src/ui/hud.py
    - Create floating, frameless, always-on-top Tkinter window
    - Implement start() to create window, update_state() to change display
    - Implement show_feedback() for action feedback messages
    - Implement process_events() for non-blocking event processing
    - Position in corner, configure opacity and size
    - _Requirements: 15.1, 15.6, 15.7_
  
  - [x] 16.2 Implement HUD state display
    - Display current interaction state: "Idle" and "Navigation Active"
    - Display feedback messages: "Next Tab", "Previous Tab", "Back", "Forward", "Tab Closed"
    - Show feedback for configurable duration
    - _Requirements: 15.2, 15.3, 15.4, 15.5_
  
  - [x] 16.3 Write unit tests for HUD
    - Test state display updates
    - Test feedback message display and expiry
    - Verify non-blocking behavior
    - _Requirements: 15.2, 15.3, 15.4, 15.5_

- [x] 17. Implement main pipeline integration
  - [x] 17.1 Create main application loop in main.py
    - Initialize all components with loaded configuration
    - Implement main loop: camera → detector → features → recognizer → state machine → resolver → dispatcher
    - Handle frame timing and loop rate
    - Integrate HUD event processing in main loop
    - _Requirements: 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 16.1_
  
  - [x] 17.2 Implement error handling in main pipeline
    - Handle camera failure (pause processing, poll for recovery)
    - Handle no-hand detection (reset state machine)
    - Handle low confidence (suppress actions)
    - Handle MediaPipe errors (continue to next frame)
    - _Requirements: 1.3, 1.4, 14.1, 14.2_
  
  - [x] 17.3 Wire HUD to pipeline events
    - Subscribe HUD to GestureEvents and BrowserIntents
    - Update HUD state on navigation engage/release
    - Show feedback on tab-switch, navigation, and close-tab intents
    - _Requirements: 15.2, 15.3, 15.4, 15.5_

- [x] 18. Write integration tests
  - [x] 18.1 Write pipeline integration tests in tests/integration/test_pipeline.py
    - Test complete frame-to-intent flow with mocked camera and MediaPipe
    - Test Victory engage → scroll → open-palm release flow
    - Test swipe detection and browser navigation flow
    - Test open-palm detection and tab close flow
    - _Requirements: 1.2, 2.2, 3.1, 4.1, 5.1, 6.1, 8.1, 9.1, 10.1, 11.1, 12.1, 13.1_
  
  - [ ]* 18.2 Write action dispatch integration tests
    - Test intent dispatch with mocked adapter
    - Verify complete gesture-to-action flow
    - Test error conditions don't produce actions
    - _Requirements: 16.1, 16.2, 16.3, 16.4, 16.5_
  
  - [ ]* 18.3 Write HUD integration tests
    - Test HUD updates on gesture events
    - Test HUD feedback on browser intents
    - _Requirements: 15.2, 15.3, 15.4, 15.5_

- [x] 19. Final checkpoint - Feature complete
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- All gesture recognition runs locally with no cloud APIs
- The design maintains strict separation between vision processing and action execution

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2", "1.3"] },
    { "id": 1, "tasks": ["2.1", "3.1", "4.1"] },
    { "id": 2, "tasks": ["2.2", "3.2", "4.2", "4.3", "4.4"] },
    { "id": 3, "tasks": ["5.1", "5.2", "6.1", "7.1"] },
    { "id": 4, "tasks": ["5.3", "5.4", "6.2", "6.3", "7.2"] },
    { "id": 5, "tasks": ["8.1", "8.2", "9.1", "9.2"] },
    { "id": 6, "tasks": ["8.3", "8.4", "9.3", "9.4", "9.5", "10.1"] },
    { "id": 7, "tasks": ["10.2", "10.3"] },
    { "id": 8, "tasks": ["12.1", "12.2", "12.3", "12.4"] },
    { "id": 9, "tasks": ["12.5", "12.6", "12.7", "12.8", "13.1", "14.1"] },
    { "id": 10, "tasks": ["13.2", "14.2"] },
    { "id": 11, "tasks": ["16.1", "16.2"] },
    { "id": 12, "tasks": ["16.3", "17.1", "17.2", "17.3"] },
    { "id": 13, "tasks": ["18.1", "18.2", "18.3"] }
  ]
}
```
