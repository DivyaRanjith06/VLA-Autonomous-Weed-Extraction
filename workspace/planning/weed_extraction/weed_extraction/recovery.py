"""The recovery tree around the scripted expert: bounded retries, every attempt labelled.

An attempt is one `Uprooter` run. It fails in one of four ways, each decided from the
same pre-action state the recorder records, so a retry never looks ahead:

- grasp_failure: the twist is over and the stem is not held, root intact. Checked there
  and not sooner, since a good grasp is taken during the twist, not the grip.
- extraction_failure: the lift is over, the stem is held and the root still holds.
- tray_drop_failure: the sequence ran to the end, but the weed is not in the bin.
- abandoned: the twist is over, the stem is not held and the root has already given
  way -- the weed was knocked loose, and there is nothing left to grasp.

A grasp failure is retried higher up the stalk; an extraction failure with more twist.
A tray-drop failure and an abandoned attempt end the episode: re-picking a loose weed
needs a sequence the expert does not have (see RecoveryPolicy.tray_drop_retries).
"""

import dataclasses
import math

from weed_extraction.uproot import Uprooter


@dataclasses.dataclass(frozen=True)
class RecoveryPolicy:
    # Where each grasp attempt takes the stalk, metres above the soil. The first is the
    # expert's own; each retry is 4 mm higher on the rigid lower 40 mm of the stalk,
    # clear of fingertip pads that settle into the soil of a sloped bed.
    grasp_heights: tuple[float, ...] = (0.016, 0.020, 0.024)
    extraction_retries: int = 1
    # The extraction retry twists harder, still below the ~15 degrees at which the twist
    # alone shears the root.
    extraction_twist: float = math.radians(13)
    # Kept for the plan's tray-drop retry; nothing can re-pick a dropped weed yet.
    tray_drop_retries: int = 0
    max_attempts: int = 3

    def __post_init__(self):
        if self.tray_drop_retries:
            raise NotImplementedError("tray-drop retries need a re-pick sequence the "
                                      "expert does not have yet")


@dataclasses.dataclass
class Attempt:
    index: int
    first_frame: int
    grasp_height_m: float
    twist_rad: float
    last_frame: int | None = None  # inclusive
    outcome: str | None = None  # success, or one of the failures in the module docstring
    detail: str = ""
    peak_load: float = 0.0
    peak_pull_n: float = 0.0
    peak_twist_nm: float = 0.0


class RetryUprooter(Uprooter):
    """The expert with the grasp height and twist of one attempt."""

    def __init__(self, env, weed, ik, grasp_height, twist):
        # Per instance: they vary by attempt. Read by Uprooter.__init__ and its phases.
        self.GRASP_HEIGHT, self.TWIST = grasp_height, twist
        super().__init__(env, weed, ik=ik)


class Recovery:
    """Runs attempts on one weed and keeps the record of each.

    `runner` is the attempt in progress. Each tick, `failure()` is asked about the state
    before the action; on a failure, `fail()` closes the attempt and, if the policy
    allows, starts the next, whose first action is that same tick's. When the runner
    finishes, `finish()` decides between success and a tray-drop failure.
    """

    def __init__(self, env, weed, ik, policy):
        self.env, self.weed, self.ik, self.policy = env, weed, ik, policy
        self.root = env.roots.of_body[env.model.body(weed).id]
        self.attempts = []
        self._start(0, policy.grasp_heights[0], Uprooter.TWIST)

    @property
    def current(self):
        return self.attempts[-1]

    def failure(self):
        """(outcome, detail) if the attempt has just failed, else None."""
        runner, data = self.runner, self.env.data
        if runner.phase >= len(runner.phases) or runner.elapsed:
            return None  # failures are only judged as a phase ends
        just_ended = runner.phases[runner.phase - 1][0] if runner.phase else None
        held = self.env.grasp.held == self.weed
        rooted = bool(data.eq_active[self.root])
        if just_ended == "twist" and not held:
            if rooted:
                return "grasp_failure", "stem not held at the end of the twist, root intact"
            return "abandoned", "the weed came loose without being grasped"
        if just_ended == "lift" and held and rooted:
            return ("extraction_failure",
                    f"still rooted at the end of the lift, peak load {runner.peak_load:.2f}")
        return None

    def fail(self, outcome, detail, frame):
        """Close the running attempt as `outcome` at `frame`; start the next if the policy
        allows. True when a new attempt is running."""
        self._close(outcome, detail, frame)
        policy, done = self.policy, self.attempts
        grasp_failures = sum(a.outcome == "grasp_failure" for a in done)
        extraction_failures = sum(a.outcome == "extraction_failure" for a in done)
        height, twist = self.current.grasp_height_m, self.current.twist_rad
        if len(done) >= policy.max_attempts:
            return False
        if outcome == "grasp_failure" and grasp_failures < len(policy.grasp_heights):
            height = policy.grasp_heights[grasp_failures]
        elif outcome == "extraction_failure" and extraction_failures <= policy.extraction_retries:
            twist = policy.extraction_twist
        else:
            return False
        # Let go of whatever the failed attempt held and restart the roots' overload
        # clocks; the weed stays where the failed attempt left it.
        self.env.grasp.let_go(self.env.data)
        self.env.roots.forget()
        self._start(frame, height, twist)
        return True

    def finish(self, frame, landed):
        """Close the last attempt once its sequence has run: `landed` is whether the weed
        ended inside the bin."""
        if landed:
            self._close("success", "", frame)
        else:
            self._close("tray_drop_failure", "released in the drop but not inside the bin",
                        frame)

    def _start(self, frame, height, twist):
        self.attempts.append(Attempt(len(self.attempts), frame, height, twist))
        self.runner = RetryUprooter(self.env, self.weed, self.ik, height, twist)

    def _close(self, outcome, detail, frame):
        attempt, runner = self.current, self.runner
        attempt.last_frame = frame - 1
        attempt.outcome, attempt.detail = outcome, detail
        attempt.peak_load = runner.peak_load
        attempt.peak_pull_n, attempt.peak_twist_nm = runner.peak_pull, runner.peak_twist
