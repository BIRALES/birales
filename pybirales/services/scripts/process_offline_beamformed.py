from matplotlib.widgets import Slider
from matplotlib import pyplot as plt
import numpy as np

data = np.load("tau_raw_processed.npy")

start_ra = -0
stop_ra = 1
delta_ra = 1
start_dec = -0
stop_dec = 1
delta_dec = 1


pointings_ra = np.arange(start_ra, stop_ra, delta_ra)
pointings_dec = np.arange(start_dec, stop_dec, delta_dec)

pointing_index = []
for i in pointings_ra:
    for j in pointings_dec:
        pointing_index.append((i, j))

print(np.where(data == np.max(data)))

#for i in range(10):
#    plt.plot(data[163-1*i, :].T)
#    plt.plot(data[163+1*i, :].T)

with open("tau_beam.txt", 'w') as f:
    f.write('\r\n'.join([str(x) for x in data.T]))


plt.plot(10*np.log(data.T))
plt.legend()
plt.show()
exit()

data = data.reshape((len(pointings_ra), len(pointings_dec), data.shape[1]))

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
