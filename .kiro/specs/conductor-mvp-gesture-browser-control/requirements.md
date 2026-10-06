# Requirements Document

## Introduction

Conductor MVP is a local, privacy-first gesture interface for browser navigation. The system captures hand gestures through a standard webcam, processes them entirely locally using MediaPipe, and translates recognized gestures into browser navigation actions. The system prioritizes prevention of accidental actions, clear visual feedback, and strict separation between vision processing and action execution.

## Glossary

- **Action_Dispatcher**: The component responsible for executing browser actions based on semantic intents.
- **Browser_Action_Adapter**: The component that translates semantic intents into OS/browser-specific actions (e.g., PyAutoGUI calls).
- **Navigation_Mode**: The control mode entered by holding a Victory/peace gesture; hand movement then controls scrolling and tab switching.
- **Navigation_Engage**: The state transition when the held Victory gesture enters navigation mode.
- **Navigation_Active**: The state while navigation mode remains enabled.
- **Navigation_Release**: The state transition when an open palm is held while navigation mode is active.
- **Deadzone**: A configurable range of motion that is ignored to prevent unintended actions from small movements.
- **Feature_Extractor**: The component that computes normalized geometric features from hand landmarks.
- **Frame_Confidence**: A measure of tracking quality for a single camera frame.
- **Gesture_Candidate**: A potential gesture recognized by the Gesture_Recognizer, pending state machine processing.
- **Gesture_Event**: A semantic event produced by the Gesture_State_Machine representing a meaningful gesture lifecycle transition or confirmed discrete gesture.
- **Gesture_Recognizer**: The component that classifies gesture types from normalized features.
- **Gesture_State_Machine**: The component that tracks gesture lifecycle and produces Gesture_Events.
- **Hand_Landmark_Detector**: The component that processes camera frames through MediaPipe to extract hand landmarks.
- **Hysteresis**: A technique using different thresholds for state transitions to prevent rapid toggling between states.
- **Intent_Resolver**: The component that maps Gesture_Events to semantic browser intents.
- **Victory_Gesture**: The two-finger gesture held to enter navigation mode.
- **Normalized_Features**: Geometric measurements scaled relative to hand size, independent of camera distance.
- **Re_arm**: A requirement for the user to return to a neutral state before a discrete gesture can trigger again.
- **Swipe_Gesture**: A ballistic horizontal hand movement used for browser history navigation.
- **Close_Gesture**: An open-palm gesture held while navigation mode is idle, used to close the current browser tab.
- **HUD**: The floating heads-up display that communicates current interaction state.

## Requirements

### Requirement 1: Camera Frame Capture

**User Story:** As a user, I want the system to capture camera frames reliably, so that gesture recognition can function.

#### Acceptance Criteria

1. WHEN the application starts, THE Camera SHALL begin capturing frames from the default webcam
2. WHEN a frame is captured, THE Camera SHALL provide the frame to the Hand_Landmark_Detector
3. IF the camera is unavailable, THEN THE Camera SHALL signal an error and THE System SHALL continue operating without crashing
4. IF camera capture fails during operation, THEN THE Camera SHALL signal an error and THE System SHALL cease gesture processing until the camera recovers

### Requirement 2: Hand Landmark Detection

**User Story:** As a user, I want the system to detect my hand landmarks accurately, so that my gestures can be recognized.

#### Acceptance Criteria

1. WHEN a camera frame is received, THE Hand_Landmark_Detector SHALL process the frame using MediaPipe to detect hand landmarks
2. WHEN hand landmarks are detected, THE Hand_Landmark_Detector SHALL output landmark coordinates and a Frame_Confidence score
3. IF no hand is detected in the frame, THE Hand_Landmark_Detector SHALL output a no-hand signal
4. IF Frame_Confidence falls below a configurable threshold, THE Hand_Landmark_Detector SHALL output a low-confidence signal
5. THE Hand_Landmark_Detector SHALL process frames entirely locally without cloud API calls

### Requirement 3: Feature Extraction

**User Story:** As a system, I need normalized geometric features from hand landmarks, so that gesture recognition is independent of camera distance.

#### Acceptance Criteria

1. WHEN hand landmarks are received, THE Feature_Extractor SHALL compute normalized geometric features
2. THE Feature_Extractor SHALL normalize all distance measurements relative to a hand-scale reference (e.g., palm size or hand bounding box)
3. THE Feature_Extractor SHALL compute pinch distance between thumb tip and middle finger tip
4. THE Feature_Extractor SHALL compute hand openness (finger extension state)
5. THE Feature_Extractor SHALL compute hand position for tracking movement over time

### Requirement 4: Navigation Mode Gesture Recognition

**User Story:** As a user, I want to enter navigation mode by holding a Victory gesture, so that I can control scrolling and tab switching.

#### Acceptance Criteria

1. WHEN a Victory gesture is held steadily, THE Gesture_Recognizer SHALL output a Victory candidate
2. THE Gesture_Recognizer SHALL apply temporal stability before the candidate is consumed by the state machine
3. THE Gesture_Recognizer SHALL use hand-landmark geometry and normalized measurements for pose classification

### Requirement 5: Navigation Mode State Machine

**User Story:** As a user, I want navigation mode to have a clear lifecycle, so that navigation control is predictable.

#### Acceptance Criteria

1. WHEN a stable Victory candidate is held for the required duration, THE Gesture_State_Machine SHALL transition from IDLE to navigation-engaged and emit a navigation-engage event
2. WHILE in navigation-engaged or navigation-active state and receiving continued Victory candidates, THE Gesture_State_Machine SHALL maintain navigation mode without emitting a repeated engage event
3. WHEN a stable open-palm candidate is held while navigation mode is active, THE Gesture_State_Machine SHALL transition to IDLE, emit a navigation-release event, and stop navigation control
4. THE Gesture_State_Machine SHALL track hand position from navigation engagement for navigation delta calculation

### Requirement 6: Scroll Control via Navigation Mode

**User Story:** As a user, I want to scroll by moving my hand vertically while navigation mode is active, so that I can navigate page content.

#### Acceptance Criteria

1. WHILE navigation mode is active, THE Intent_Resolver SHALL track the current and previous valid hand positions and calculate incremental vertical movement between them
2. WHEN incremental vertical movement exceeds the configurable deadzone, THE Intent_Resolver SHALL emit a scroll intent proportional to the incremental displacement, and a stationary hand SHALL NOT continuously generate additional scroll intents
3. WHEN incremental vertical movement is upward beyond the configurable deadzone, THE Intent_Resolver SHALL emit a scroll-up intent
4. WHEN incremental vertical movement is downward beyond the configurable deadzone, THE Intent_Resolver SHALL emit a scroll-down intent
5. THE Intent_Resolver SHALL use hand-scale-normalized displacement measurements
6. THE Action_Dispatcher SHALL execute scroll actions when receiving scroll intents

### Requirement 7: Tab Switch Control via Navigation Mode

**User Story:** As a user, I want to switch tabs by moving my hand horizontally while navigation mode is active, so that I can navigate between browser tabs.

#### Acceptance Criteria

1. WHILE navigation mode is active, THE Intent_Resolver SHALL track horizontal hand displacement
2. WHEN horizontal displacement exceeds a configurable threshold to the right, THE Intent_Resolver SHALL emit a next-tab intent
3. WHEN horizontal displacement exceeds a configurable threshold to the left, THE Intent_Resolver SHALL emit a previous-tab intent
4. AFTER emitting a tab-switch intent, THE Intent_Resolver SHALL require the hand to return to within a re-arm threshold before another tab-switch intent can be emitted
5. THE Intent_Resolver SHALL use hand-scale-normalized displacement measurements
6. THE Action_Dispatcher SHALL execute tab-switch actions when receiving tab-switch intents

### Requirement 8: Ballistic Swipe Gesture Recognition

**User Story:** As a user, I want to navigate browser history by swiping horizontally, so that I can go back or forward without navigation mode.

#### Acceptance Criteria

1. WHEN navigation mode is not active, THE Gesture_Recognizer SHALL analyze horizontal hand movement over multiple frames
2. THE Gesture_Recognizer SHALL evaluate swipe motion over a bounded sequence of frames using net displacement, elapsed time, velocity, and horizontal direction consistency
3. WHEN a horizontal movement within the bounded frame sequence meets the configurable velocity and displacement thresholds and maintains sufficient directional consistency, THE Gesture_Recognizer SHALL output a swipe candidate with direction (left or right)
4. THE Gesture_Recognizer SHALL reject movements below the displacement threshold
5. THE Gesture_Recognizer SHALL reject movements below the velocity threshold
6. THE Gesture_Recognizer SHALL reject movements whose frame-to-frame horizontal direction is sufficiently inconsistent with the overall swipe direction
7. THE Gesture_Recognizer SHALL use hand-scale-normalized measurements for thresholds

### Requirement 9: Swipe Gesture State Machine

**User Story:** As a user, I want swipe gestures to trigger exactly one browser action, so that history navigation is predictable.

#### Acceptance Criteria

1. WHEN a swipe candidate is received, THE Gesture_State_Machine SHALL validate the candidate against temporal criteria
2. WHEN a valid swipe-right candidate is confirmed, THE Gesture_State_Machine SHALL emit a Swipe_Right event
3. WHEN a valid swipe-left candidate is confirmed, THE Gesture_State_Machine SHALL emit a Swipe_Left event
4. AFTER emitting a swipe event, THE Gesture_State_Machine SHALL require the hand to return to a neutral position before another swipe can be recognized
5. THE Gesture_State_Machine SHALL reject swipe candidates when Frame_Confidence is below the configurable threshold

### Requirement 10: Browser History Navigation via Swipe

**User Story:** As a user, I want swipe gestures to navigate browser history, so that I can go back and forward.

#### Acceptance Criteria

1. WHEN a Swipe_Right event is received, THE Intent_Resolver SHALL emit a browser-forward intent
2. WHEN a Swipe_Left event is received, THE Intent_Resolver SHALL emit a browser-back intent
3. THE Action_Dispatcher SHALL execute browser forward action when receiving a browser-forward intent
4. THE Action_Dispatcher SHALL execute browser back action when receiving a browser-back intent

### Requirement 11: Open Palm Close Gesture Recognition

**User Story:** As a user, I want to close the current tab by holding an open palm while idle, so that I can close tabs without an unreliable fist pose.

#### Acceptance Criteria

1. WHEN all five fingers are extended in an open-palm configuration and navigation mode is idle, THE Gesture_Recognizer SHALL output an open-palm candidate
2. THE Gesture_Recognizer SHALL use hand-scale-normalized measurements to determine the open-palm configuration
3. THE Gesture_Recognizer SHALL NOT emit a closed-fist candidate from live camera input

### Requirement 12: Open Palm Close Gesture State Machine

**User Story:** As a user, I want the open-palm close gesture to require deliberate holding, so that accidental tab closures are prevented.

#### Acceptance Criteria

1. WHEN an open-palm candidate is first detected while navigation mode is idle, THE Gesture_State_Machine SHALL transition from NEUTRAL to a close candidate state
2. WHEN the open-palm candidate continues for a configurable hold duration, THE Gesture_State_Machine SHALL confirm the close gesture
3. WHEN the close gesture is confirmed, THE Gesture_State_Machine SHALL emit a Tab_Close event
4. AFTER emitting a Tab_Close event, THE Gesture_State_Machine SHALL transition to WAIT_FOR_REARM
5. WHEN the open-palm gesture is released, THE Gesture_State_Machine SHALL transition from WAIT_FOR_REARM to NEUTRAL
6. THE Gesture_State_Machine SHALL NOT emit additional Tab_Close events while remaining in the confirmed or WAIT_FOR_REARM states
7. IF the open-palm gesture is released before the hold duration, THE Gesture_State_Machine SHALL return to NEUTRAL without emitting an event

### Requirement 13: Tab Close Action

**User Story:** As a user, I want the confirmed idle open-palm gesture to close the current browser tab, so that I can dismiss tabs with a gesture.

#### Acceptance Criteria

1. WHEN a Tab_Close event is received, THE Intent_Resolver SHALL emit a close-tab intent
2. THE Action_Dispatcher SHALL execute tab-close action when receiving a close-tab intent

### Requirement 14: Low Confidence and Tracking Failure Safety

**User Story:** As a user, I want the system to ignore gestures when tracking is unreliable, so that accidental actions are prevented.

#### Acceptance Criteria

1. WHEN Frame_Confidence is below a configurable threshold, THE Gesture_State_Machine SHALL suppress all gesture state transitions
2. WHEN no hand is detected, THE Gesture_State_Machine SHALL reset to neutral states
3. WHEN a tracking loss is detected (sudden large coordinate jump), THE Gesture_State_Machine SHALL suppress gesture state transitions for a configurable cooldown period
4. WHEN the time since the last valid frame exceeds a configurable staleness threshold, THE Gesture_State_Machine SHALL suppress gesture state transitions
5. THE Gesture_State_Machine SHALL never emit gesture events during suppressed state

### Requirement 15: Floating HUD Display

**User Story:** As a user, I want a floating heads-up display showing the current interaction state, so that I understand what the system is doing.

#### Acceptance Criteria

1. WHEN the application starts, THE HUD SHALL display a floating, frameless, always-on-top window
2. THE HUD SHALL display the current interaction state including: idle and navigation-active
3. WHEN a tab-switch intent is emitted, THE HUD SHALL display "Next Tab" or "Previous Tab" for a visible duration
4. WHEN a browser navigation intent is emitted, THE HUD SHALL display "Back" or "Forward" for a visible duration
5. WHEN a close-tab intent is emitted, THE HUD SHALL display "Tab Closed" for a visible duration
6. THE HUD SHALL be visually minimal and non-blocking
7. THE HUD SHALL run its event loop in the main thread

### Requirement 16: Action Dispatcher

**User Story:** As a system, I need a central dispatcher for browser actions, so that action execution is controlled and testable.

#### Acceptance Criteria

1. WHEN the Action_Dispatcher receives a scroll intent, THE Action_Dispatcher SHALL invoke the Browser_Action_Adapter to execute scrolling
2. WHEN the Action_Dispatcher receives a tab-switch intent, THE Action_Dispatcher SHALL invoke the Browser_Action_Adapter to switch tabs
3. WHEN the Action_Dispatcher receives a browser-navigation intent, THE Action_Dispatcher SHALL invoke the Browser_Action_Adapter to navigate history
4. WHEN the Action_Dispatcher receives a close-tab intent, THE Action_Dispatcher SHALL invoke the Browser_Action_Adapter to close the tab
5. THE Action_Dispatcher SHALL be the only component that invokes the Browser_Action_Adapter

### Requirement 17: Browser Action Adapter

**User Story:** As a system, I need to translate intents into browser actions, so that gestures control the browser.

#### Acceptance Criteria

1. WHEN invoked with a scroll intent, THE Browser_Action_Adapter SHALL execute vertical scroll via PyAutoGUI
2. WHEN invoked with a next-tab intent, THE Browser_Action_Adapter SHALL execute the platform-appropriate keyboard shortcut via PyAutoGUI
3. WHEN invoked with a previous-tab intent, THE Browser_Action_Adapter SHALL execute the platform-appropriate keyboard shortcut via PyAutoGUI
4. WHEN invoked with a browser-forward intent, THE Browser_Action_Adapter SHALL execute the platform-appropriate keyboard shortcut via PyAutoGUI
5. WHEN invoked with a browser-back intent, THE Browser_Action_Adapter SHALL execute the platform-appropriate keyboard shortcut via PyAutoGUI
6. WHEN invoked with a close-tab intent, THE Browser_Action_Adapter SHALL execute the close-tab keyboard shortcut via PyAutoGUI

### Requirement 18: Configuration System

**User Story:** As a user, I want all thresholds and sensitivities to be configurable, so that I can adjust the system to my preferences.

#### Acceptance Criteria

1. THE System SHALL load configuration from a configuration file at startup
2. THE System SHALL expose configurable parameters including: deadzone size, swipe velocity threshold, swipe displacement threshold, close-gesture hold duration (currently keyed as `fist_hold_duration` for compatibility), confidence threshold, staleness threshold, and tracking-jump threshold
3. WHEN configuration values are missing, THE System SHALL use documented default values

### Requirement 19: Testability Without Hardware

**User Story:** As a developer, I want to test the system without a physical camera, so that tests are reliable and fast.

#### Acceptance Criteria

1. THE Hand_Landmark_Detector SHALL accept mock frame inputs for testing
2. THE Gesture_Recognizer SHALL accept mock feature inputs for testing
3. THE Gesture_State_Machine SHALL accept mock gesture candidate inputs for testing
4. THE Intent_Resolver SHALL accept mock gesture event inputs for testing
5. THE Action_Dispatcher SHALL be mockable to verify intents without executing browser actions
