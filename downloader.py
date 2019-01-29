import pandas as pd
import requests
import os
from queue import Queue
from time import time
from threading import Thread


class DownloadWorker(Thread):
    def __init__(self, queue):
        Thread.__init__(self)
        self.queue = queue

    def run(self):
        while True:
            # Get the work from the queue and expand the tuple
            url, filename = self.queue.get()
            response = requests.get(url)

            if response.status_code == 200:
                with open(filename, 'wb') as f:
                    f.write(response.content)
            else:
                print('Error with: {}'.format(url))

            self.queue.task_done()



def main():

    print('=====================================================================================================')
    print('                                   DOWNLOAD STARTED                                                  ')
    print('=====================================================================================================')
    ts = time()

    # Prepare output folder
    base_folder = 'Data/Images/'
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)

    # Create a queue to communicate with the worker threads
    queue = Queue()

    # Read data
    data = pd.read_csv('Data/data.csv').Photo

    # Create 8 worker threads
    for x in range(20):
        worker = DownloadWorker(queue)

        # Setting daemon to True will let the main thread exit even though the workers are blocking
        worker.daemon = True
        worker.start()

    print('Total URLs: {}'.format(len(data)))

    # Put the tasks into the queue as a tuple
    for i, url in enumerate(data):
        queue.put((url, '{}/{}.png'.format(base_folder, i)))

    # Causes the main thread to wait for the queue to finish processing all the tasks
    queue.join()
    print('Took {}'.format(time() - ts))


if __name__ == '__main__':
    main()
