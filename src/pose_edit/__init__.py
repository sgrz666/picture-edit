"""Shared building blocks for the pose-transfer workflows (single and dual person).

Nothing here knows how many people a task has: cache/fingerprint rules, canvas
preprocessing, the SMPLest-X call wrapper and the DWPose-134 joint tables live
together so the single-person and two-person workflows cannot drift apart.
"""
