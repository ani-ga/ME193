"""Hardware-agnostic lock policy (state machine). Same code on laptop and UNO Q.
Feed it one (prediction, confidence) per camera frame via update()."""
import time
import config

IDLE, ARMED, UNLOCKED, LOCKOUT = "IDLE", "ARMED", "UNLOCKED", "LOCKOUT"

class LockPolicy:
    def __init__(self, on_leds=None, on_open=None, on_close=None, on_digit=None, clock=time.monotonic):
        nop = lambda *a: None
        self.on_leds, self.on_open = on_leds or nop, on_open or nop
        self.on_close, self.on_digit = on_close or nop, on_digit or nop
        self.clock = clock
        self.state, self.failed = IDLE, 0
        self._reset_entry()

    def _reset_entry(self):
        self.digits, self.streak_class, self.streak = [], None, 0
        self.need_blank = False            # after a digit, hand must go away (class 0)

    def press_button(self):
        """IDLE: arm.  ARMED (button mode): lock in the digit being shown.  UNLOCKED: re-lock.
        Returns True/False when a digit was locked in / refused, otherwise None."""
        if self.state == IDLE:
            self.state, self.t0 = ARMED, self.clock()
            self._reset_entry(); self.on_leds(True)
        elif self.state == UNLOCKED:       # button re-locks early
            self._lock()
        elif self.state == ARMED and config.BUTTON_CONFIRMS_DIGITS:
            return self._confirm_digit()

    def _confirm_digit(self):
        """Accept the number currently on show, if it has been steady for a few frames."""
        digit = self.streak_class
        if digit is None or digit == 0 or self.streak < config.BUTTON_MIN_FRAMES:
            return False                   # no clear 1-5 right now: ignore this press
        self._accept(digit)
        return True

    def _accept(self, digit):
        self.digits.append(digit)
        self.on_digit(digit, len(self.digits))
        if len(self.digits) == len(config.LOCK_CODE): self._check()

    def _lock(self):
        self.on_close(); self.state = IDLE; self.on_leds(False)

    def update(self, pred, conf):
        now = self.clock()
        if self.state == LOCKOUT and now - self.t0 >= config.LOCKOUT_SECONDS:
            self.state, self.failed = IDLE, 0
        elif self.state == UNLOCKED and now - self.t0 >= config.OPEN_SECONDS:
            self._lock()
        elif self.state == ARMED:
            if now - self.t0 >= config.TIMEOUT_SECONDS:
                self.state = IDLE; self.on_leds(False); return
            self._read_digit(pred, conf)

    def _read_digit(self, pred, conf):
        if conf < config.CONF_THRESHOLD:           # not sure -> restart the streak
            self.streak_class, self.streak = None, 0; return
        if pred == self.streak_class: self.streak += 1
        else: self.streak_class, self.streak = pred, 1
        if config.BUTTON_CONFIRMS_DIGITS: return   # button mode: just keep counting, the button decides
        if self.streak < config.STABLE_FRAMES: return
        if pred == 0:
            self.need_blank = False                # hand is away: ready for next digit
        elif not self.need_blank:
            self.need_blank = True
            self._accept(pred)
        self.streak = 0

    def _check(self):
        self.on_leds(False)
        if tuple(self.digits) == tuple(config.LOCK_CODE):
            self.state, self.t0, self.failed = UNLOCKED, self.clock(), 0
            self.on_open()
        else:
            self.failed += 1
            if self.failed >= config.MAX_ATTEMPTS: self.state, self.t0 = LOCKOUT, self.clock()
            else: self.state = IDLE
