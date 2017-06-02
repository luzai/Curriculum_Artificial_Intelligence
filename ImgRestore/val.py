import multiprocessing
import val_client

mp_queue = multiprocessing.Queue()

val_client.main(mp_queue, 'gray_denoise')
