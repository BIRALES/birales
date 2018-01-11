from matplotlib.widgets import Slider
from matplotlib import pyplot as plt
import numpy as np

data = np.load("casa_raw_processed_offset.npy")

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
with open('casa_tranist_offset.txt', 'w') as f:
     [f.write("{},{},{},{},{},{},{},{},{}\r\n".format(str(x[0]),str(x[1]),str(x[2]),str(x[3]),str(x[4]),str(x[5]),str(x[6]),str(x[7]),str(x[8]))) for x in data.T]

plt.plot(10*np.log10(data.T[:,0]), label="[-1.6, 1]")
plt.plot(10*np.log10(data.T[:,1]), label="[0, 1]")
plt.plot(10*np.log10(data.T[:,2]), label="[1.6, 1]")
plt.plot(10*np.log10(data.T[:,3]), label="[-1.6, 0]")
plt.plot(10*np.log10(data.T[:,4]), label="[0, 0]")
plt.plot(10*np.log10(data.T[:,5]), label="[1.6, 0]")
plt.plot(10*np.log10(data.T[:,6]), label="[-1.6, -1")
plt.plot(10*np.log10(data.T[:,7]), label="[0, -1]")
plt.plot(10*np.log10(data.T[:,8]), label="[1.6, -1]")
plt.legend()
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
