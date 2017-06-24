from matplotlib.widgets import Slider
from matplotlib import pyplot as plt
import numpy as np

data = np.load("virgo_raw_processed.npy")

start_ra = -5
stop_ra = 5
delta_ra = 0.5
start_dec = -5
stop_dec = 5
delta_dec = 0.5

#pointings_ra = np.arange(start_ra, stop_ra, delta_ra)
#pointings_dec = np.arange(start_dec, stop_dec, delta_dec)
#
#pointing_index = []
#for i in pointings_ra:
#    for j in pointings_dec:
#        pointing_index.append((i, j))
#
#print(np.where(data == np.max(data)))
#
#for i in range(10):
#    plt.plot(data[163-1*i, :].T)
#    plt.plot(data[163+1*i, :].T)
plt.plot(data.T)
#plt.axvline(x=8883)
plt.show()
exit()

print data.shape
data = data.reshape((len(pointings_ra), len(pointings_dec), data.shape[1]))
print data.shape

fig = plt.figure()
ax = fig.add_subplot(111)
fig.tight_layout()
fig.subplots_adjust(left=0.15, bottom=0.25)

obj = plt.imshow(data[:,:,7000].T, aspect='auto', origin='bottom')
plt.xlabel("Delta RA")
plt.ylabel("Delta DEC")

amp_slider_ax = fig.add_axes([0.25, 0.15, 0.65, 0.03])


amp_slider = Slider(amp_slider_ax, 'Amp', 0, data.shape[2] - 1 , valinit=0)

def sliders_on_changed(val):
    obj.set_data(data[:,:,int(val)].T)
    fig.canvas.draw_idle()
amp_slider.on_changed(sliders_on_changed)

plt.show()
