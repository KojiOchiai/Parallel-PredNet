import numpy
import cupy
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import variable


class EltFilter(chainer.Link):
    def __init__(self, width, height, channels, batchSize=1, wscale=1, bias=0,
                 nobias=False, initialW=None, initial_bias=None):
        W_shape = (batchSize, channels, height, width)
        super(EltFilter, self).__init__(W=W_shape)

        if initialW is not None:
            self.W.data[...] = initialW
        else:
            std = wscale * numpy.sqrt(1. / (width * height * channels))
            self.W.data[...] = numpy.random.normal(0, std, W_shape)

        if nobias:
            self.b = None
        else:
            self.add_param('b', W_shape)
            if initial_bias is None:
                initial_bias = bias
            self.b.data[...] = initial_bias

    def __call__(self, x):
        y = x * self.W
        if self.b is not None:
            y = y + self.b
        return y


class ConvLSTM(chainer.Chain):
    def __init__(self, width, height, in_channels, out_channels, batchSize=1):
        self.state_size = (batchSize, out_channels, height, width)
        self.in_channels = in_channels
        super(ConvLSTM, self).__init__(
            h_i=L.Convolution2D(out_channels, out_channels, 3, pad=1),
            c_i=EltFilter(width, height, out_channels, nobias=True),

            h_f=L.Convolution2D(out_channels, out_channels, 3, pad=1),
            c_f=EltFilter(width, height, out_channels, nobias=True),

            h_c=L.Convolution2D(out_channels, out_channels, 3, pad=1),

            h_o=L.Convolution2D(out_channels, out_channels, 3, pad=1),
            c_o=EltFilter(width, height, out_channels, nobias=True),
        )

        for nth in range(len(self.in_channels)):
            self.add_link('x_i' + str(nth),
                          L.Convolution2D(self.in_channels[nth],
                                          out_channels,
                                          3, pad=1, nobias=True))
            self.add_link('x_f' + str(nth),
                          L.Convolution2D(self.in_channels[nth],
                                          out_channels,
                                          3, pad=1, nobias=True))
            self.add_link('x_c' + str(nth),
                          L.Convolution2D(self.in_channels[nth],
                                          out_channels,
                                          3, pad=1, nobias=True))
            self.add_link('x_o' + str(nth),
                          L.Convolution2D(self.in_channels[nth],
                                          out_channels,
                                          3, pad=1, nobias=True))

        self.reset_state()

    def to_cpu(self):
        super(ConvLSTM, self).to_cpu()
        if self.c is not None:
            self.c.to_cpu()
        if self.h is not None:
            self.h.to_cpu()

    def to_gpu(self, device=None):
        super(ConvLSTM, self).to_gpu(device)
        if self.c is not None:
            self.c.to_gpu(device)
        if self.h is not None:
            self.h.to_gpu(device)

    def reset_state(self):
        self.c = self.h = None

    def __call__(self, x):
        if self.h is None:
            self.h = variable.Variable(
                self.xp.zeros(self.state_size, dtype=x[0].data.dtype),
                volatile='auto')
        if self.c is None:
            self.c = variable.Variable(
                self.xp.zeros(self.state_size, dtype=x[0].data.dtype),
                volatile='auto')

        ii = self.x_i0(x[0])
        for nth in range(1, len(self.in_channels)):
            ii += getattr(self, 'x_i' + str(nth))(x[nth])
        ii += self.h_i(self.h)
        ii += self.c_i(self.c)
        ii = F.sigmoid(ii)

        ff = self.x_f0(x[0])
        for nth in range(1, len(self.in_channels)):
            ff += getattr(self, 'x_f' + str(nth))(x[nth])
        ff += self.h_f(self.h)
        ff += self.c_f(self.c)
        ff = F.sigmoid(ff)

        cc = self.x_c0(x[0])
        for nth in range(1, len(self.in_channels)):
            cc += getattr(self, 'x_c' + str(nth))(x[nth])
        cc += self.h_c(self.h)
        cc = F.tanh(cc)
        cc *= ii
        cc += (ff * self.c)

        oo = self.x_o0(x[0])
        for nth in range(1, len(self.in_channels)):
            oo += getattr(self, 'x_o' + str(nth))(x[nth])
        oo += self.h_o(self.h)
        oo += self.c_o(self.c)
        oo = F.sigmoid(oo)
        y = oo * F.tanh(cc)

        self.c = cc
        self.h = y
        return y


class PredLayer(chainer.Chain):
    def __init__(self, width, height, channels,
                 r_channels=None, batchSize=1, pooling=2,
                 istop=False, isbottom=False):
        self.istop = istop
        self.isbottom = isbottom
        self.device = None
        if r_channels is None:
            r_channels = channels
        self.sizes = [None]*2
        w, h = width, height
        for nth in range(len(channels)):
            self.sizes[nth] = (batchSize, channels[nth], h, w)
            w = w / pooling
            h = h / pooling

        if self.istop:
            super(PredLayer, self).__init__(
                    ConvP=L.Convolution2D(r_channels[0],
                                          channels[0],
                                          3,
                                          pad=1),
                    ConvLSTM=ConvLSTM(self.sizes[0][3],
                                      self.sizes[0][2],
                                      (self.sizes[0][1] * 2,),
                                      r_channels[0])
            )
        else:
            super(PredLayer, self).__init__(
                    ConvA=L.Convolution2D(channels[0] * 2,
                                          channels[1],
                                          3,
                                          pad=1),
                    ConvP=L.Convolution2D(r_channels[0],
                                          channels[0],
                                          3,
                                          pad=1),
                    ConvLSTM=ConvLSTM(self.sizes[0][3],
                                      self.sizes[0][2],
                                      (self.sizes[0][1] * 2,
                                       r_channels[1]),
                                      r_channels[0])
            )
        self.reset_state()

    def to_cpu(self):
        super(PredLayer, self).to_cpu()
        self.P.to_cpu()

    def to_gpu(self, device=None):
        self.device = device
        super(PredLayer, self).to_gpu(device)
        self.P.to_gpu(device)

    def reset_state(self):
        self.P = variable.Variable(
                 self.xp.zeros(self.sizes[0], dtype=numpy.float32),
                 volatile='auto')
        self.ConvLSTM.reset_state()

    def __call__(self, bottom_up, top_down=None):
        with cupy.cuda.Device(self.device):
            E = F.concat((F.relu(bottom_up - self.P),
                          F.relu(self.P - bottom_up)))
            if self.istop:
                A = None
                R = self.ConvLSTM((E,))
            else:
                A = F.max_pooling_2d(F.relu(self.ConvA(E)), 2, stride=2)
                unpooled = F.unpooling_2d(top_down, 2,
                                          stride=2, cover_all=False)
                R = self.ConvLSTM((E, unpooled))

            if self.isbottom:
                P = F.clipped_relu(self.ConvP(R), 1.0)
            else:
                P = F.relu(self.ConvP(R))

        self.P = P

        return (A, R)


class PredNet(chainer.Chain):
    def __init__(self, width, height, channels, r_channels=None, batchSize=1,
                 devices=None):
        super(PredNet, self).__init__()

        self.layers = len(channels)
        if isinstance(devices, int) or devices is None:
            self.devices = [devices] * self.layers
        else:
            self.devices = devices
        self.sizes = [None]*self.layers
        w, h = width, height
        for nth in range(self.layers):
            self.sizes[nth] = (batchSize, channels[nth], h, w)
            w = w / 2
            h = h / 2

        for nth in range(self.layers):
            if nth == self.layers - 1:
                self.add_link('Layer' + str(nth),
                              PredLayer(self.sizes[nth][3],
                                        self.sizes[nth][2],
                                        (self.sizes[nth][1],),
                                        istop=True))
            else:
                self.add_link('Layer' + str(nth),
                              PredLayer(self.sizes[nth][3],
                                        self.sizes[nth][2],
                                        (self.sizes[nth][1],
                                         self.sizes[nth + 1][1]),
                                        isbottom=nth == 0))
        self.reset_state()

    def to_cpu(self):
        super(PredNet, self).to_cpu()
        for nth in range(self.layers - 1):
            getattr(self, 'A' + str(nth + 1)).to_cpu()
            getattr(self, 'R' + str(nth)).to_cpu()
            getattr(self, 'Layer' + str(nth)).to_cpu()
        nth = self.layers - 1
        getattr(self, 'Layer' + str(nth)).to_cpu()

    def to_gpu(self, device=None):
        if device is None:
            super(PredNet, self).to_gpu(device)
            if not isinstance(self.devices[0], int):
                self.devices = [device] * self.layers
        elif isinstance(device, int):
            super(PredNet, self).to_gpu(device)
            self.devices = [device] * self.layers
        else:
            super(PredNet, self).to_gpu(device[0])
            self.devices = device

        for nth in range(self.layers - 1):
            getattr(self, 'A' + str(nth + 1)).to_gpu(self.devices[nth + 1])
            getattr(self, 'R' + str(nth)).to_gpu(self.devices[nth])
            getattr(self, 'Layer' + str(nth)).to_gpu(self.devices[nth])
        nth = self.layers - 1
        getattr(self, 'Layer' + str(nth)).to_gpu(self.devices[nth])

    def reset_state(self):
        for nth in range(self.layers - 1):
            setattr(self, 'A' + str(nth + 1),
                    variable.Variable(
                    self.xp.zeros(self.sizes[nth + 1], dtype=numpy.float32),
                    volatile='auto'))
            setattr(self, 'R' + str(nth),
                    variable.Variable(
                    self.xp.zeros(self.sizes[nth + 1], dtype=numpy.float32),
                    volatile='auto'))
            getattr(self, 'Layer' + str(nth)).reset_state()
        getattr(self, 'Layer' + str(self.layers - 1)).reset_state()

    def __call__(self, x):
        A = [None] * self.layers
        R = [None] * (self.layers - 1)

        # update layers
        for nth in range(self.layers):
            if nth == 0 & nth == self.layers - 1:
                (_, _) = self.Layer0(x)
            elif nth == 0:
                (A[1], _) = self.Layer0(x, self.R0)
            elif nth == self.layers - 1:
                (_, R[nth - 1]) = getattr(self, 'Layer' + str(nth))(
                                    getattr(self, 'A' + str(nth)))
            else:
                (A[nth + 1], R[nth - 1]) = getattr(self, 'Layer' + str(nth))(
                                          getattr(self, 'A' + str(nth)),
                                          getattr(self, 'R' + str(nth)))

        # copy data to device
        if isinstance(x.data, numpy.ndarray) or self.devices[nth] is None:
            for nth in range(self.layers - 1):
                setattr(self, 'A' + str(nth + 1), A[nth + 1])
                setattr(self, 'R' + str(nth), R[nth])
        else:
            for nth in range(self.layers - 1):
                setattr(self, 'A' + str(nth + 1),
                        F.copy(A[nth + 1], self.devices[nth + 1]))
                setattr(self, 'R' + str(nth),
                        F.copy(R[nth], self.devices[nth]))

        return self.Layer0.P
