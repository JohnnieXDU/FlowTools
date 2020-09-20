import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch


class FlowTools:
    def __init__(self):
        self.colorwheel = self._make_colorwheel()

    @staticmethod
    def _make_colorwheel():
        """
        Generates a color wheel for optical flow visualization as presented in:
            Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
            URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

        Code follows the original C++ source code of Daniel Scharstein.
        Code follows the the Matlab source code of Deqing Sun.

        Returns:
            np.ndarray: Color wheel
        """

        RY = 15
        YG = 6
        GC = 4
        CB = 11
        BM = 13
        MR = 6

        ncols = RY + YG + GC + CB + BM + MR
        colorwheel = np.zeros((ncols, 3))
        col = 0

        # RY
        colorwheel[0:RY, 0] = 255
        colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
        col = col+RY
        # YG
        colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
        colorwheel[col:col+YG, 1] = 255
        col = col+YG
        # GC
        colorwheel[col:col+GC, 1] = 255
        colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
        col = col+GC
        # CB
        colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
        colorwheel[col:col+CB, 2] = 255
        col = col+CB
        # BM
        colorwheel[col:col+BM, 2] = 255
        colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
        col = col+BM
        # MR
        colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
        colorwheel[col:col+MR, 0] = 255
        return colorwheel  # shape [55x3]

    def _flow_uv_to_colors(self, u, v, convert_to_bgr=False):
        """
        Applies the flow color wheel to (possibly clipped) flow components u and v.

        According to the C++ source code of Daniel Scharstein
        According to the Matlab source code of Deqing Sun

        Args:
            u (np.ndarray): Input horizontal flow of shape [H,W]
            v (np.ndarray): Input vertical flow of shape [H,W]
            convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

        Returns:
            np.ndarray: Flow visualization image of shape [H,W,3]
        """
        flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
        ncols = self.colorwheel.shape[0]
        rad = np.sqrt(np.square(u) + np.square(v))
        a = np.arctan2(-v, -u)/np.pi
        fk = (a+1) / 2*(ncols-1)
        k0 = np.floor(fk).astype(np.int32)
        k1 = k0 + 1
        k1[k1 == ncols] = 0
        f = fk - k0
        for i in range(self.colorwheel.shape[1]):
            tmp = self.colorwheel[:,i]
            col0 = tmp[k0] / 255.0
            col1 = tmp[k1] / 255.0
            col = (1-f)*col0 + f*col1
            idx = (rad <= 1)
            col[idx]  = 1 - rad[idx] * (1-col[idx])
            col[~idx] = col[~idx] * 0.75   # out of range
            # Note the 2-i => BGR instead of RGB
            ch_idx = 2-i if convert_to_bgr else i
            flow_image[:,:,ch_idx] = np.floor(255 * col)
        return flow_image

    def _flow_to_image(self, flow_uv, clip_flow=None, convert_to_bgr=False):
        """
        Expects a two dimensional flow image of shape.

        Args:
            flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
            clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
            convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

        Returns:
            np.ndarray: Flow visualization image of shape [H,W,3]
        """
        assert flow_uv.ndim == 3, 'input flow must have three dimensions'
        assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
        if clip_flow is not None:
            flow_uv = np.clip(flow_uv, 0, clip_flow)
        u = flow_uv[:,:,0]
        v = flow_uv[:,:,1]
        rad = np.sqrt(np.square(u) + np.square(v))
        rad_max = np.max(rad)
        epsilon = 1e-5
        u = u / (rad_max + epsilon)
        v = v / (rad_max + epsilon)
        return self._flow_uv_to_colors(u, v, convert_to_bgr)

    def _format_check(self, flow):
        """convert flow format to np.array with size [H, W, 2]."""

        if isinstance(flow, torch.Tensor):
            flow = flow.cpu().detach().numpy()

        if isinstance(flow, np.ndarray):
            flow = flow.astype(np.float)  # check 32-bit for mixed-precision training

            if not len(np.shape(flow)) == 3:
                raise Exception('Flow size should be 3-dims [H, W, 2] or [2, H, W].')

            if flow.shape[0] == 2:
                flow = flow.transpose(1, 2, 0)

        else:
            raise Exception('Unrecognized format.')

        return flow

    def vizflow(self, flow, viz=False):
        """ Visualize optical flow.
        """
        flow = self._format_check(flow)
        flow_rgb = self._flow_to_image(flow)

        if viz:
            plt.figure()
            plt.imshow(flow_rgb)
            plt.show()

        return flow_rgb

    def epemap(self, flow, flow_gt):
        """ Compute EPE map between flow and flow_gt.
        """

        flow = self._format_check(flow)
        flow_gt = self._format_check(flow_gt)

        # L-2 norm Euclidean EPE
        diff = (flow_gt - flow).astype(np.float)
        errmap = np.sqrt(np.sum(diff**2, axis=2))
        epe = errmap.mean()

        return errmap, epe