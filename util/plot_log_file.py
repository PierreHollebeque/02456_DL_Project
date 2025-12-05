import matplotlib.pyplot as plt
import re

data = {}

with open('loss_log.txt', 'r') as f:
    for line in f:
        if 'epoch:' not in line:
            continue
        
        matches = re.findall(r'([a-zA-Z_]+): ([\d\.]+)', line)
        for key, value in matches:
            if key not in data:
                data[key] = []
            data[key].append(float(value))

exclude = ['epoch', 'iters', 'time', 'data']
plot_keys = [k for k in data.keys() if k not in exclude]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

for key in plot_keys:
    if 'cycle' in key or 'idt' in key:
        ax2.plot(data[key], label=key)
    else:
        ax1.plot(data[key], label=key)

ax1.set_title('GAN and Discriminator Losses')
ax1.legend()
ax1.grid(True)

ax2.set_title('Cycle and Identity Losses')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()