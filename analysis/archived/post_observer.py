
def post_observable(method):
    observers = []

    def singleton_per_method(self, *args, **kwargs):
        method(self, *args, **kwargs)
        done = []
        for i, fun in enumerate(observers):
            if fun(*args, **kwargs):
                done.append(i)
        done.sort(reverse=True)
        for i in done:
            del observers[i]

    class has_addObserver(Protocol):
        """just an interface, the real functions are just below"""

        def addObserver(self, val):
            ...

        def __call__(self, *args, **kwargs):
            ...

    def addObserver(observer):
        observers.append(observer)

    singleton_per_method.addObserver = addObserver
    singleton_per_method: has_addObserver
    return singleton_per_method
