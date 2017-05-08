from matplotlib.widgets import Slider
from matplotlib import pyplot as plt
import numpy as np

data = np.load("casa_raw_processed_on_source.npy")

pointings_ra = np.arange(-10, 10.5, 0.5)
pointings_dec = np.arange(-10, 10.5, 0.5)

pointing_index = []
size = len(np.arange(-10, 10.5, 10))
for i in np.arange(-10, 10.5, 0.5):
    for j in np.arange(-10, 10.5, 0.5):
        pointing_index.append((i, j))

print(np.where(data == np.max(data)))
print(pointing_index[218])

# plt.plot(data[217, :].T)
# for i in range(10):
#     plt.plot(data[217-41*i, :].T)
#     plt.plot(data[217+41*i, :].T)
# plt.show()

print data.shape
data = data.reshape((len(pointings_ra), len(pointings_dec), data.shape[1]))
print data.shape

fig = plt.figure()
ax = fig.add_subplot(111)
fig.tight_layout()
fig.subplots_adjust(left=0.15, bottom=0.25)

obj = plt.imshow(data[:,:,7000].T, aspect='auto', origin='bottom', extent=[-10, 10, -10, 10])
plt.xlabel("Delta RA")
plt.ylabel("Delta DEC")

amp_slider_ax = fig.add_axes([0.25, 0.15, 0.65, 0.03])


amp_slider = Slider(amp_slider_ax, 'Amp', 0, data.shape[2] - 1 , valinit=0)

def sliders_on_changed(val):
    obj.set_data(data[:,:,int(val)].T)
    fig.canvas.draw_idle()
amp_slider.on_changed(sliders_on_changed)

plt.show()