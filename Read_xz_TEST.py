import numpy as np
import lzma
import time

start = time.time()

#f = lzma.open("/mnt/Resources/immortal/Model_Photospheres/BT-Settl_M-0.0a+0.0/lte028.0-4.5-0.0a+0.0.BT-Settl.spec.7.xz") # 44 seconds
f = lzma.open("lte028.0-4.5-0.0a+0.0.BT-Settl.spec.7.xz") # 45 seconds

Model_wave, Model_Flux, Model_BB_Flux = np.loadtxt(f, usecols=(0,1,2), unpack=True, converters={1: lambda s: float(str(s).split('D')[0].strip('b').replace("'","")) * 10**int(str(s).split('D')[1].replace("'","")), 2: lambda s: float(str(s).split('D')[0].strip('b').replace("'","")) * 10**int(str(s).split('D')[1].replace("'",""))})

print(time.time() - start)

