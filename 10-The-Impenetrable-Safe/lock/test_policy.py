"""Run: python -m unittest lock.test_policy"""
import unittest
import config
from lock.policy import LockPolicy, IDLE, ARMED, UNLOCKED, LOCKOUT

class Clock:
    t = 0.0
    def __call__(self): return self.t

def hold(p, digit, conf=0.99):
    for _ in range(config.STABLE_FRAMES): p.update(digit, conf)

def enter(p, code):
    for d in code:
        hold(p, d); hold(p, 0)

def show_and_press(p, digit, conf=0.99, frames=None):
    """Button mode: show a digit for a few frames, then press the button."""
    for _ in range(frames or config.BUTTON_MIN_FRAMES): p.update(digit, conf)
    return p.press_button()

class T(unittest.TestCase):
    """Hold-to-accept mode (button only arms)."""
    def setUp(self):
        config.BUTTON_CONFIRMS_DIGITS = False
        self.c = Clock(); self.log = []
        self.p = LockPolicy(on_open=lambda: self.log.append("open"),
                            on_close=lambda: self.log.append("close"), clock=self.c)
    def test_correct_code_opens(self):
        self.p.press_button(); enter(self.p, config.LOCK_CODE)
        self.assertEqual(self.p.state, UNLOCKED); self.assertEqual(self.log, ["open"])
    def test_wrong_code_stays_locked(self):
        self.p.press_button(); enter(self.p, (1, 1 if config.LOCK_CODE[1] != 1 else 2, 4))
        self.assertEqual(self.p.state, IDLE); self.assertEqual(self.log, [])
    def test_held_digit_counts_once(self):
        self.p.press_button(); hold(self.p, 2); hold(self.p, 2); hold(self.p, 2)
        self.assertEqual(self.p.digits, [2])
    def test_low_confidence_ignored(self):
        self.p.press_button(); hold(self.p, 2, conf=0.5)
        self.assertEqual(self.p.digits, [])
    def test_unstable_ignored(self):
        self.p.press_button()
        for i in range(40): self.p.update(1 + i % 2, 0.99)
        self.assertEqual(self.p.digits, [])
    def test_timeout(self):
        self.p.press_button(); self.c.t = config.TIMEOUT_SECONDS + 1; self.p.update(0, 0.99)
        self.assertEqual(self.p.state, IDLE)
    def test_lockout_after_max_attempts(self):
        bad = (1, 2, 4) if config.LOCK_CODE != (1, 2, 4) else (1, 2, 5)
        for _ in range(config.MAX_ATTEMPTS):
            self.p.press_button(); enter(self.p, bad)
        self.assertEqual(self.p.state, LOCKOUT)
        self.p.press_button(); self.assertEqual(self.p.state, LOCKOUT)
    def test_auto_relock(self):
        self.p.press_button(); enter(self.p, config.LOCK_CODE)
        self.c.t = config.OPEN_SECONDS + 1; self.p.update(0, 0.99)
        self.assertEqual(self.log, ["open", "close"])

class ButtonMode(unittest.TestCase):
    """Button locks in each digit; hand does not need to leave."""
    def setUp(self):
        config.BUTTON_CONFIRMS_DIGITS = True
        self.c = Clock(); self.log = []
        self.p = LockPolicy(on_open=lambda: self.log.append("open"), clock=self.c)
        self.p.press_button()               # arm
    def tearDown(self): config.BUTTON_CONFIRMS_DIGITS = True
    def test_correct_code_opens(self):
        for d in config.LOCK_CODE: self.assertTrue(show_and_press(self.p, d))
        self.assertEqual(self.p.state, UNLOCKED); self.assertEqual(self.log, ["open"])
    def test_no_need_to_drop_hand(self):          # same digit twice in a row is allowed
        show_and_press(self.p, 2); show_and_press(self.p, 2)
        self.assertEqual(self.p.digits, [2, 2])
    def test_wrong_code_stays_locked(self):
        wrong = (1, 1, 4) if config.LOCK_CODE != (1, 1, 4) else (1, 1, 5)
        for d in wrong: show_and_press(self.p, d)
        self.assertEqual(self.p.state, IDLE); self.assertEqual(self.log, [])
    def test_no_hand_press_ignored(self):
        self.assertFalse(show_and_press(self.p, 0)); self.assertEqual(self.p.digits, [])
    def test_unsteady_press_ignored(self):
        self.assertFalse(show_and_press(self.p, 3, frames=1)); self.assertEqual(self.p.digits, [])
    def test_unsure_press_ignored(self):
        self.assertFalse(show_and_press(self.p, 3, conf=0.4)); self.assertEqual(self.p.digits, [])
    def test_changing_number_needs_new_streak(self):
        for _ in range(10): self.p.update(2, 0.99)
        self.p.update(4, 0.99)                    # just switched to 4: not steady yet
        self.assertFalse(self.p.press_button())
    def test_timeout(self):
        self.c.t = config.TIMEOUT_SECONDS + 1; self.p.update(2, 0.99)
        self.assertEqual(self.p.state, IDLE)

if __name__ == "__main__": unittest.main()
