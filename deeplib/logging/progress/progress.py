from time import time


class progress:
    """
    TODO: Add description.

    Heavily inspired by tqdm: https://github.com/tqdm/tqdm
    """

    def __init__(
        self,
        iterable=None,
        total=None,
        mininterval=0.1,
        maxinterval=10.0,
        initial=0,
        miniters=None,
        disable=False,
    ):
        if total is None and iterable is not None:
            try:
                total = len(iterable)
            except (TypeError, AttributeError):
                total = None
        if total == float("inf"):
            total = None

        if disable:
            self.iterable = iterable
            self.disable = disable
            self.n = initial
            self.total = total
            return

        if miniters is None:
            miniters = 0

        if mininterval is None:
            mininterval = 0

        if maxinterval is None:
            maxinterval = 0

        self.iterable = iterable
        self.total = total
        self.n = initial
        self.mininterval = mininterval
        self.maxinterval = maxinterval
        self.miniters = miniters
        self._time = time

    def __len__(self):
        return (
            self.total
            if self.iterable is None
            else self.iterable.shape[0]
            if hasattr(self.iterable, "shape")
            else len(self.iterable)
            if hasattr(self.iterable, "__len__")
            else self.iterable.__length_hint__()
            if hasattr(self.iterable, "__length_hint__")
            else getattr(self, "total", None)
        )

    def __iter__(self):
        iterable = self.iterable

        mininterval = self.mininterval
        last_print_t = self.last_print_t
        last_print_n = self.last_print_n
        min_start_t = self.start_t + self.delay
        n = self.n
        time = self._time

        try:
            for obj in iterable:
                yield obj

                n += 1

                if n - last_print_n >= self.miniters:
                    cur_t = time()
                    dt = cur_t - last_print_t
                    if dt >= mininterval and cur_t >= min_start_t:
                        self.update(n - last_print_n)
                        last_print_n = self.last_print_n
                        last_print_t = self.last_print_t
        finally:
            self.n = n
            self.close()

    def update(self, n=1):
        if self.disable:
            return

        if n < 0:
            self.last_print_n += n
        self.n += n

        if self.n - self.last_print_n >= self.miniters:
            cur_t = self._time()
            dt = cur_t - self.last_print_t
            if dt >= self.mininterval and cur_t >= self.start_t + self.delay:
                cur_t = self._time()
                dn = self.n - self.last_print_n  # >= n

                self.refresh(lock_args=self.lock_args)

                # Store old values for next call
                self.last_print_n = self.n
                self.last_print_t = cur_t
                return True

    def close(self):
        """Cleanup and (if leave=False) close the progressbar."""
        if self.disable:
            return

        # Prevent multiple closures
        self.disable = True

        # decrement instance pos and remove from internal set
        pos = abs(self.pos)
        self._decr_instances(self)

        if self.last_print_t < self.start_t + self.delay:
            # haven't ever displayed; nothing to clear
            return

        # GUI mode
        if getattr(self, "sp", None) is None:
            return

        # annoyingly, _supports_unicode isn't good enough
        def fp_write(s):
            self.fp.write(_unicode(s))

        try:
            fp_write("")
        except ValueError as e:
            if "closed" in str(e):
                return
            raise  # pragma: no cover

        leave = pos == 0 if self.leave is None else self.leave

        with self._lock:
            if leave:
                # stats for overall rate (no weighted average)
                self._ema_dt = lambda: None
                self.display(pos=0)
                fp_write("\n")
            else:
                # clear previous display
                if self.display(msg="", pos=pos) and not pos:
                    fp_write("\r")

    def refresh(self, nolock=False, lock_args=None):
        """
        Force refresh the display of this bar.

        Parameters
        ----------
        nolock  : bool, optional
            If `True`, does not lock.
            If [default: `False`]: calls `acquire()` on internal lock.
        lock_args  : tuple, optional
            Passed to internal lock's `acquire()`.
            If specified, will only `display()` if `acquire()` returns `True`.
        """
        if self.disable:
            return

        if not nolock:
            if lock_args:
                if not self._lock.acquire(*lock_args):
                    return False
            else:
                self._lock.acquire()
        self.display()
        if not nolock:
            self._lock.release()
        return True

    def display(self, msg=None, pos=None):
        """
        Use `self.sp` to display `msg` in the specified `pos`.

        Consider overloading this function when inheriting to use e.g.:
        `self.some_frontend(**self.format_dict)` instead of `self.sp`.

        Parameters
        ----------
        msg  : str, optional. What to display (default: `repr(self)`).
        pos  : int, optional. Position to `moveto`
          (default: `abs(self.pos)`).
        """
        if pos is None:
            pos = abs(self.pos)

        nrows = self.nrows or 20
        if pos >= nrows - 1:
            if pos >= nrows:
                return False
            if msg or msg is None:  # override at `nrows - 1`
                msg = " ... (more hidden) ..."

        if pos:
            self.moveto(pos)
        self.sp(self.__str__() if msg is None else msg)
        if pos:
            self.moveto(-pos)
        return True
