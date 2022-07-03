import cv2 as cv
import numpy as np
from progress.bar import IncrementalBar

from .estimation.models import EstimationModelFactory


class FileEstimator:
    def __init__(self, model_type='seresnet18', split_four=True, verbose=1, batch_size=4):
        self.model = EstimationModelFactory().get_estimation_model(model_type)
        self.batch_generator = VideoFileBatchGenerator(
            target_shape=self.model.get_input_shape(),
            split_four=split_four,
            verbose=verbose, 
            batch_size=batch_size,
            return_last_batch=True
        )
    
    def evaluate(self, file_path):
        self.batch_generator.set_file_path(file_path)
        total_evaluations = []
        for batch in self.batch_generator.get_batches():
            total_evaluations.extend(self.model.predict(np.array(batch)))
        return np.average(total_evaluations)


class VideoFileBatchGenerator:
    def __init__(self, target_shape, batch_size=4, split_four=True, verbose=1, return_last_batch=True):
        self.file_path = None
        self.cap = None
        self.batch_size = batch_size
        self.transformers = []
        if split_four:
            self.transformers.append(self._transform_split_four)
        if target_shape[3] == 1:
            self.transformers.append(self._transform_to_grayscale)
        self.verbose = verbose
        self.width = target_shape[1]
        self.height = target_shape[2]
        self.frames_len = target_shape[0]
        self.color_channels = target_shape[3]
        self.return_last_batch = return_last_batch

    def set_file_path(self, file_path):
        self.file_path = file_path
        self.cap = cv.VideoCapture(self.file_path)
        if not self.cap.isOpened():
            raise FileNotFoundError('File not found: {}'.format(self.file_path))

    def get_batches(self):
        assert self.file_path is not None
        if self.verbose > 0:
            print('Loading file: {}'.format(self.file_path))
            print('Frame count: {}'.format(int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))))
            print('Frame width: {}'.format(int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))))
            print('Frame height: {}'.format(int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))))
            print('Frame rate: {}'.format(int(self.cap.get(cv.CAP_PROP_FPS))))
            bar = IncrementalBar('Countdown', max = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT)) // self.frames_len)
        all_frames = []
        while True:
            frames = []
            for i in range(self.frames_len):
                ret, frame = self.cap.read()
                if not ret:
                    break
                frames.append(frame)
            else:
                if self.verbose > 0:
                    bar.next()
            if len(frames) != self.frames_len:
                break
            all_frames.extend(self._transform([frames]))
            del frames
            while len(all_frames) >= self.batch_size:
                yield all_frames[:self.batch_size]
                all_frames = all_frames[self.batch_size:]
        self.cap.release()
        if self.verbose > 0:
            bar.finish()
        if self.return_last_batch and len(all_frames) > 0:
            yield all_frames

    def _transform(self, frames_batch):
        for transformer in self.transformers:
            frames_batch = transformer(frames_batch)
        return frames_batch
    
    def _transform_split_four(self, frames_batch):
        assert len(frames_batch[0]) == self.frames_len
        batches_len = len(frames_batch)
        width = np.shape(frames_batch[0])[1]
        height = np.shape(frames_batch[0])[2]
        square_side = min(width//2, height//2)
        result = np.empty((batches_len * 4, self.frames_len, self.width, self.height, np.shape(frames_batch[0])[3]), dtype=np.uint8)
        
        for j, frames in enumerate(frames_batch):
            for i in range(self.frames_len):
                cropped = frames[i][
                        (width//2) - square_side:(width//2),
                        (height//2) - square_side:(height//2),
                    ]
                result[j, i] = cv.resize(cropped, (self.width, self.height))
                cropped = frames[i][
                        (width//2):(width//2) + square_side,
                        (height//2):(height//2) + square_side,
                    ]
                result[j + 1, i] = cv.resize(cropped, (self.width, self.height))
                cropped = frames[i][
                        (width//2) - square_side:(width//2),
                        (height//2):(height//2) + square_side,
                    ]
                result[j + 2, i] = cv.resize(cropped, (self.width, self.height))
                cropped = frames[i][
                        (width//2):(width//2) + square_side,
                        (height//2) - square_side:(height//2),
                    ]
                result[j + 3, i] = cv.resize(cropped, (self.width, self.height))
        return result
    
    def _transform_to_grayscale(self, frames_batch):
        for i in range(len(frames_batch)):
            for j in range(len(frames_batch[i])):
                frames_batch[i][j] = cv.cvtColor(frames_batch[i][j], cv.COLOR_BGR2GRAY)
        
        return frames_batch
