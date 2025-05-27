import torch as th
import numpy as np
from PIL import Image
# pytorch=1.7.1
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
# pip install opencv-python
import cv2
import os
from PIL import Image

class RawVideoExtractorCV2():
    def __init__(self, centercrop=False, size=224, framerate=-1, ):
        self.centercrop = centercrop
        self.size = size
        self.framerate = framerate
        self.transform = self._transform(self.size)

    # def _transform(self, n_px):
    #     return Compose([
    #         Resize((n_px,n_px), interpolation=Image.BICUBIC),
    #         # CenterCrop(n_px),
    #         lambda image: image.convert("RGB"),
    #         ToTensor(),
    #         # Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    #     ])

    def _transform(self, n_px):
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def video_to_tensor(self, video_file, preprocess, sample_fp=0, start_time=None, end_time=None):
        if start_time is not None or end_time is not None:
            assert isinstance(start_time, int) and isinstance(end_time, int) \
                   and start_time > -1 and end_time > start_time
        assert sample_fp > -1

        # Samples a frame sample_fp X frames.
        # cap = cv2.VideoCapture(video_file)
        # frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # fps = int(cap.get(cv2.CAP_PROP_FPS))

        frames = [file for file in os.listdir(video_file) if 'jpg' in file]
        frames_len = len(frames)


        def _load_image(directory, idx):
            return Image.open(os.path.join(directory, f"{idx:04d}.jpg")).convert('RGB')

        self.random_sample = False
        if self.random_sample:
            # print(f'DEBUGJM: {path} {len(frames)}')
            frame_idx = self._random_sample_frame_idx(frames_len)
            frames = [preprocess(_load_image(video_file, x+1)) for x in frame_idx]
            # frames = [frames[x].to_rgb().to_ndarray() for x in frame_idx]
            # frames = th.as_tensor(np.stack(frames)).float() / 255.
        else:
            frames = [preprocess(_load_image(video_file, x+1)) for x in range(frames_len)] #preprocess(
            frames = th.as_tensor(np.stack(frames))

        
        return {'video': frames}
        # total_duration = len(frames) # (frameCount + fps - 1) // fps
        # start_sec, end_sec = 0, total_duration

        # if start_time is not None:
        #     start_sec, end_sec = start_time, end_time if end_time <= total_duration else total_duration
        #     # cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_time * fps))

        # # interval = 1
        # # if sample_fp > 0:
        # #     interval = fps // sample_fp
        # # else:
        # #     sample_fp = fps
        # # if interval == 0: interval = 1

        # sample_fp = 8
        # interval = 1
        # fps = 8

        # inds = [ind for ind in np.arange(0, fps, interval)]
        # assert len(inds) >= sample_fp
        # inds = inds[:sample_fp]

        # ret = True
        # images, included = [], []

        # for sec in np.arange(start_sec, end_sec + 1):
        #     if not ret: break
        #     sec_base = int(sec * fps)
        #     for ind in inds:
        #         cap.set(cv2.CAP_PROP_POS_FRAMES, sec_base + ind)
        #         ret, frame = cap.read()
        #         if not ret: break
        #         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #         images.append(preprocess(Image.fromarray(frame_rgb).convert("RGB")))

        # cap.release()

        # if len(images) > 0:
        #     video_data = th.tensor(np.stack(images))
        # else:
        #     video_data = th.zeros(1)

    def _random_sample_frame_idx(self, len):
        frame_indices = []
        # print(f'DEBUGJM: {len}')
        if self.sampling_rate <= 0: # tsn sample
            seg_size = (len - 1) / self.num_frames
            for i in range(self.num_frames):
                start, end = round(seg_size * i), round(seg_size * (i + 1))
                frame_indices.append(np.random.randint(start, end + 1)) # random
        elif self.sampling_rate * (self.num_frames - 1) + 1 >= len:
            for i in range(self.num_frames):
                frame_indices.append(i * self.sampling_rate if i * self.sampling_rate < len else frame_indices[-1])
        else:
            start = np.random.randint(len - self.sampling_rate * (self.num_frames - 1))
            frame_indices = list(range(start, start + self.sampling_rate * self.num_frames, self.sampling_rate))

        return frame_indices

    def get_video_data(self, video_path, start_time=None, end_time=None):
        image_input = self.video_to_tensor(video_path, self.transform, sample_fp=self.framerate, start_time=start_time, end_time=end_time)
        return image_input

    def process_raw_data(self, raw_video_data):
        tensor_size = raw_video_data.size()
        tensor = raw_video_data.view(-1, 1, tensor_size[-3], tensor_size[-2], tensor_size[-1])
        return tensor

    def process_frame_order(self, raw_video_data, frame_order=0):
        # 0: ordinary order; 1: reverse order; 2: random order.
        if frame_order == 0:
            pass
        elif frame_order == 1:
            reverse_order = np.arange(raw_video_data.size(0) - 1, -1, -1)
            raw_video_data = raw_video_data[reverse_order, ...]
        elif frame_order == 2:
            random_order = np.arange(raw_video_data.size(0))
            np.random.shuffle(random_order)
            raw_video_data = raw_video_data[random_order, ...]

        return raw_video_data

# An ordinary video frame extractor based CV2
RawVideoExtractor = RawVideoExtractorCV2