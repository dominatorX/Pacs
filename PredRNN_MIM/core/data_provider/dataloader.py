import pandas as pd
import numpy as np
import bisect
from core.data_provider import image
import os


Image_Path = '/export/data/civil_proj/HKO_radar_images/radarPNG'


def get_exclude_mask():
    with np.load('../data/mask_dat.npz') as dat:
        exclude_mask = dat['exclude_mask'][:]
        return exclude_mask


def convert_datetime_to_filepath(date_time):
    """Convert datetime to the filepath

    Parameters
    ----------
    date_time : datetime.datetime

    Returns
    -------
    ret : str
    """
    ret = os.path.join("%04d" % date_time.year,
                       "%02d" % date_time.month,
                       "%02d" % date_time.day,
                       'RAD%02d%02d%02d%02d%02d00.png'
                       % (date_time.year - 2000, date_time.month, date_time.day,
                          date_time.hour, date_time.minute))
    ret = os.path.join(Image_Path, ret)
    return ret


_exclude_mask = get_exclude_mask()


class HKOIterator(object):
    def __init__(self, pd_path, sample_mode, configs, seq_len=30,
                 max_consecutive_missing=0, begin_ind=None, end_ind=None,
                 stride=None, base_freq='6min'):
        """Random sample: sample a random clip that will not violate the max_missing frame_num criteria
        Sequent sample: sample a clip from the beginning of the time.
                        Everytime, the clips from {T_begin, T_begin + 6min, ..., T_begin + (seq_len-1) * 6min} will be used
                        The begin datetime will move forward by adding stride: T_begin += 6min * stride
                        Once the clips violates the maximum missing number criteria, the starting
                         point will be moved to the next datetime that does not violate the missing_frame criteria

        Parameters
        ----------
        pd_path : str
            path of the saved pandas dataframe
        sample_mode : str
            Can be "random" or "sequent"
        seq_len : int
        max_consecutive_missing : int
            The maximum consecutive missing frames
        begin_ind : int
            Index of the begin frame
        end_ind : int
            Index of the end frame
        stride : int or None, optional
        width : int or None, optional
        height : int or None, optional
        base_freq : str, optional
        """
        self._df = pd.read_pickle(pd_path)
        self._df_all = pd.read_pickle(configs.RAINY_ALL)
        self.set_begin_end(begin_ind=begin_ind, end_ind=end_ind)
        self._df_index_set_all = frozenset([self._df_all.index[i] for i in range(self._df_all.size)])
        self._exclude_mask = 1-get_exclude_mask()[np.newaxis, 170: 332, 102: 315]
        self._channel = configs.img_channel
        self._seq_len = seq_len
        self._width = configs.img_width
        self._height = configs.img_height
        self._stride = stride
        self._max_consecutive_missing = max_consecutive_missing
        self._base_freq = base_freq
        self._base_time_delta = pd.Timedelta(base_freq)

        assert sample_mode in ["random", "sequent"], "Sample mode=%s is not supported" % sample_mode
        self.sample_mode = sample_mode
        if sample_mode == "sequent":
            assert self._stride is not None
            self._current_datetime = self.begin_time

    def set_begin_end(self, begin_ind=None, end_ind=None):
        self._begin_ind = 0 if begin_ind is None else begin_ind
        self._end_ind = self.total_frame_num - 1 if end_ind is None else end_ind

    @property
    def total_frame_num(self):
        return self._df.size

    @property
    def begin_time(self):
        return self._df.index[self._begin_ind]

    @property
    def end_time(self):
        return self._df.index[self._end_ind]

    @property
    def use_up(self):
        if self.sample_mode == "random":
            return False
        else:
            return self._current_datetime > self.end_time

    def _next_exist_timestamp(self, timestamp):
        next_ind = bisect.bisect_right(self._df.index, timestamp)
        if next_ind >= self._df.size:
            return None
        else:
            return self._df.index[bisect.bisect_right(self._df.index, timestamp)]

    def _is_valid_clip(self, datetime_clip):
        """Check if the given datetime_clip is valid

        Parameters
        ----------
        datetime_clip :

        Returns
        -------
        ret : bool
        """
        missing_count = 0
        if datetime_clip is None:
            return False
        for i in range(len(datetime_clip)):
            if datetime_clip[i] not in self._df_index_set_all:
                # print(datetime_clip[i])
                missing_count += 1
                if missing_count > self._max_consecutive_missing or \
                        missing_count >= len(datetime_clip):
                    return False
            else:
                missing_count = 0
        return True

    def _load_frames(self, datetime_clips):
        assert isinstance(datetime_clips, list)
        for clip in datetime_clips:
            assert len(clip) == self._seq_len
        batch_size = len(datetime_clips)

        frame_dat = np.zeros((self._seq_len, batch_size, 5 if self._channel != 1 else 1,
                              self._height, self._width), dtype=np.uint8 if self._channel == 1 else np.float32)
        mask_dat = np.zeros((self._seq_len, batch_size, 1, self._height, self._width), dtype=np.bool)
        if batch_size == 0:
            return frame_dat, mask_dat
        if self.sample_mode == "random":
            paths, hit_inds, miss_inds = [], [], []
            for i in range(batch_size):
                for j in range(self._seq_len):
                    timestamp = datetime_clips[i][j]
                    if timestamp in self._df_index_set_all:
                        paths.append(convert_datetime_to_filepath(timestamp))
                        hit_inds.append([j, i])
                    else:
                        miss_inds.append([j, i])

            all_mask_dat = self._exclude_mask[np.newaxis, ...].repeat(self._seq_len*batch_size, axis=0)
            hit_inds = np.array(hit_inds, dtype=np.int)
            all_frame_dat = image.quick_read_frames(path_list=paths,
                                                    im_h=self._height,
                                                    im_w=self._width,
                                                    channel=self._channel)
            frame_dat[hit_inds[:, 0], hit_inds[:, 1], :, :, :] = all_frame_dat
            mask_dat[hit_inds[:, 0], hit_inds[:, 1], :, :, :] = all_mask_dat

        else:
            count = 0
            pending_key = {}
            paths = []

            for i in range(self._seq_len):
                for j in range(batch_size):
                    timestamp = datetime_clips[j][i]
                    if timestamp in self._df_index_set_all and timestamp not in pending_key:
                        pending_key[timestamp] = count
                        count += 1
                        paths.append(convert_datetime_to_filepath(timestamp))

            pending_frame = image.quick_read_frames(path_list=paths,
                                                    im_h=self._height,
                                                    im_w=self._width,
                                                    channel=self._channel)
            pending_mask = self._exclude_mask
            for i in range(self._seq_len):
                for j in range(batch_size):
                    timestamp = datetime_clips[j][i]
                    if timestamp in self._df_index_set_all:
                        assert timestamp in pending_key
                        ind = pending_key[timestamp]
                        frame_dat[i, j, :, :, :] = pending_frame[ind, :, :, :]
                        mask_dat[i, j, :, :, :] = pending_mask
        if self._channel == 1:
            data_batch = frame_dat.astype(np.float32) / 255.0
        else:
            data_batch = frame_dat

            # [0.021, 22.794, 32.968, 45.279]
            # [0.004, -2.474, -56.426, -43.313]
            data_batch[:, :, 4, :, :] /= 255.0
            data_batch[:, :, 3, :, :] = (data_batch[:, :, 3, :, :] - 0.004) / 0.017
            data_batch[:, :, 2, :, :] = (data_batch[:, :, 2, :, :] + 2.474) / 25.268
            data_batch[:, :, 1, :, :] = (data_batch[:, :, 1, :, :] + 56.426) / 89.394
            data_batch[:, :, 0, :, :] = (data_batch[:, :, 0, :, :] + 43.313) / 88.592
            if self._channel == 3:
                data_batch = np.concatenate([np.sqrt(np.square(data_batch[:, :, 0:1, ...]) +
                                                     np.square(data_batch[:, :, 1:2, ...])),
                                             data_batch[:, :, 2:3, ...], data_batch[:, :, 4:5, ...]], axis=2)

        return data_batch, mask_dat

    def reset(self, begin_ind=None, end_ind=None):
        assert self.sample_mode == "sequent"
        self.set_begin_end(begin_ind=begin_ind, end_ind=end_ind)
        self._current_datetime = self.begin_time

    def random_reset(self):
        assert self.sample_mode == "sequent"
        self.set_begin_end(begin_ind=np.random.randint(0,
                                                       self.total_frame_num -
                                                       5 * self._seq_len),
                           end_ind=None)
        self._current_datetime = self.begin_time

    def check_new_start(self):
        assert self.sample_mode == "sequent"
        datetime_clip = pd.date_range(start=self._current_datetime,
                                      periods=self._seq_len,
                                      freq=self._base_freq)
        if self._is_valid_clip(datetime_clip):
            return self._current_datetime == self.begin_time
        else:
            return True

    def sample(self, batch_size, only_return_datetime=False):
        """
        Sample a minibatch from the hko7 dataset based on the given type and pd_file
        Parameters
        ----------
        batch_size : int
            Batch size
        only_return_datetime : bool
            Whether to only return the datetimes
        Returns
        -------
        frame_dat : np.ndarray
            Shape: (seq_len, valid_batch_size, 1, height, width)
        mask_dat : np.ndarray
            Shape: (seq_len, valid_batch_size, 1, height, width)
        datetime_clips : list
            length should be valid_batch_size
        new_start : bool
        """
        datetime_clip_p1, datetime_clip_p2 = None, None

        if self.sample_mode == 'sequent':
            if self.use_up:
                raise ValueError("The HKOIterator has been used up!")
            datetime_clips = []
            new_start = False
            for i in range(batch_size):
                while not self.use_up:
                    datetime_clip = pd.date_range(start=self._current_datetime,
                                                  periods=self._seq_len,
                                                  freq=self._base_freq)

                    if self._is_valid_clip(datetime_clip):

                        new_start = new_start or (self._current_datetime == self.begin_time)
                        datetime_clips.append(datetime_clip)
                        self._current_datetime += self._stride * self._base_time_delta
                        break
                    else:
                        new_start = True
                        self._current_datetime = \
                            self._next_exist_timestamp(timestamp=self._current_datetime)
                        if self._current_datetime is None:
                            # This indicates that there is no timestamp left,
                            # We point the current_datetime to be the next timestamp of self.end_time
                            self._current_datetime = self.end_time + self._base_time_delta
                            break
                        continue
            new_start = None if batch_size != 1 else new_start
            if only_return_datetime:
                return datetime_clips, new_start
        else:
            assert only_return_datetime is False
            datetime_clips = []
            new_start = None
            for i in range(batch_size):
                while True:
                    rand_ind = np.random.randint(0, self._df.size, 1)[0]
                    random_datetime = self._df.index[rand_ind]

                    datetime_clip = pd.date_range(start=random_datetime,
                                                  periods=self._seq_len,
                                                  freq=self._base_freq)

                    if self._is_valid_clip(datetime_clip):
                        datetime_clips.append(datetime_clip)
                        break
        frame_dat, mask_dat = self._load_frames(datetime_clips=datetime_clips)
        return frame_dat, mask_dat, datetime_clips, new_start
