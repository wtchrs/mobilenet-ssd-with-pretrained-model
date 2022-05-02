import queue, threading
import cv2


class BufferlessVideoCapture:
    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self._reader)
        self.thread.daemon = True
        self.thread.start()

    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.queue.empty():
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    pass
            self.queue.put(frame)

    def read(self):
        return self.queue.get()
