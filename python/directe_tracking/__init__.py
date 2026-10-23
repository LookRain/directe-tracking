"""Directe's climber tracker, the Python reference implementation (the Swift package in Sources/ is a bit-exact
port, checked by the golden tests).

  detect        YOLO pose on whole frames (Ultralytics), cached per clip
  identity      re-ID embeddings (boxmot OSNet) on person crops, masked or plain
  appearance    colour signatures by body region
  motion        is the wall moving here (second chance for a missed climber)
  camera        whole-image shift between ticks (phase correlation) for hand-held video
  registration  how the climber is tagged at the start (a tap, or a pose)
  tracker       TargetTracker: who the climber is, tick to tick (Locked / Uncertain / Lost)
  online        the whole loop on a clip at 10 fps, as a phone runs it
  labels        hand-checked identity labels, what tracker changes are judged on
  clips, video, pipeline   the clip registry, decoding, model-set selection
"""
