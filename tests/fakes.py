"""Playwright stand-ins for the tests.

They implement only the methods the code under test actually uses and make no attempt
to reproduce Playwright. Their point is to turn **timing problems** such as "listeners
not unbound across tasks" or "closing pages down to 0" into assertable state: bugs that
are hardest to reproduce in a real browser and most likely to blow up in a real run.
"""


class FakeRequest:
    def __init__(self, url, resource_type="xhr", method="GET", post_data=None):
        self.url = url
        self.method = method
        self.headers = {"user-agent": "test"}
        self.resource_type = resource_type
        self.post_data_buffer = post_data


class FakeElement:
    """<select> stand-in: records the selected index for assertions."""

    def __init__(self):
        self.selected_index = None

    def select_option(self, index=None):
        self.selected_index = index


class FakeHandle:
    def __init__(self, element):
        self._element = element

    def as_element(self):
        return self._element


class FakePage:
    def __init__(self, name="page", context=None, close_error=None,
                 element=None, options=None):
        self.name = name
        self.context = context
        self.closed = False
        self.close_error = close_error          # simulate "this page cannot be closed"
        self.stopped = False                    # whether window.stop() was called
        self._handlers = {}
        self.element = element                  # the <select> hit by elementFromPoint
        self.options = options or []            # its options
        self.waited_ms = 0
        self.url = "https://example.test/"
        # excute_action reads the viewport and the page fingerprint before and after the action (no-op detection)
        self.viewport_size = {"width": 1920, "height": 1080}
        self.fingerprint = "fp-0"
        if context is not None:
            context.pages.append(self)

    # --- events ---
    def on(self, event, handler):
        self._handlers.setdefault(event, []).append(handler)

    def remove_listener(self, event, handler):
        if self.closed:
            raise RuntimeError("Page has been closed")   # real Playwright may raise too
        self._handlers.get(event, []).remove(handler)

    def listener_count(self, event="request"):
        return len(self._handlers.get(event, []))

    def emit_request(self, request):
        """Simulate the page sending a network request. A closed page must not send any."""
        assert not self.closed, f"{self.name} is closed and should not send requests"
        for handler in list(self._handlers.get("request", [])):
            handler(request)

    # --- lifecycle ---
    def close(self, run_before_unload=False):
        if self.close_error:
            raise RuntimeError(self.close_error)
        self.closed = True
        if self.context is not None and self in self.context.pages:
            self.context.pages.remove(self)

    def is_closed(self):
        return self.closed

    # --- evaluation ---
    def evaluate(self, script, arg=None):
        if "window.stop" in script:
            self.stopped = True
            return None
        if "el.options" in script:
            return self.options
        if "elementFromPoint" in script:
            return None
        return self.fingerprint          # what page_fingerprint returns

    def evaluate_handle(self, script, arg=None):
        return FakeHandle(self.element)

    def wait_for_timeout(self, ms):
        self.waited_ms += ms

    def wait_for_load_state(self, state=None, timeout=None):
        pass


class FakeContext:
    def __init__(self):
        self.pages = []

    def new_page(self):
        return FakePage(name=f"page{len(self.pages)}", context=self)
