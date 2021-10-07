
import time
import math
from pvaccess import Channel
from pvaccess import PvObject
    
class pvaClient:
    def __init__(self, ):
        pass

    def monitor(self, pv):
        uid = pv['uniqueId']
        print("received frame %d @ %.3f" % (uid, time.time()))
        rows = pv['attribute'][0]['value'][0]['value']
        cols = pv['attribute'][1]['value'][0]['value']
        frame = pv['value'][0]['ushortValue'].reshape((rows, cols))
        print(frame.shape)
        time.sleep(.8) # emulating data processing

def main_monitor():
    max_queue_size = -1
    c = Channel('pvapy:image')
    c.setMonitorMaxQueueLength(max_queue_size)

    client = pvaClient()

    c.subscribe('monitor', client.monitor)
    c.startMonitor('')

    time.sleep(1000)
    c.stopMonitor()

    c.unsubscribe('monitor')

def main_get():
    max_queue_size = 10
    c = Channel('pvapy:image')
    c.setMonitorMaxQueueLength(max_queue_size)

    while True:
        pv = c.get()
        frame = pv['value'][0]['ushortValue']
        print(frame.shape)


if __name__ == '__main__':
    main_monitor()
    # main_get()

