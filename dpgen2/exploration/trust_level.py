class TrustLevel():
    def __init__(
            self,
            level_f_lo,
            level_f_hi,
            level_v_lo = None,
            level_v_hi = None,
    ):
        self._level_f_lo = level_f_lo
        self._level_f_hi = level_f_hi
        self._level_v_lo = level_v_lo
        self._level_v_hi = level_v_hi

    @property
    def level_f_lo(self):
        return self._level_f_lo

    @property
    def level_f_hi(self):
        return self._level_f_hi

    @property
    def level_v_lo(self):
        return self._level_v_lo

    @property
    def level_v_hi(self):
        return self._level_v_hi
