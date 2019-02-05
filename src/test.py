import threading
import time

def f1():
    i = 0
    while True:
        print(i)
        i += 5
        time.sleep(2)

def f2():
    j = 0
    while True:
        print(j)
        j += 7
        time.sleep(3)
        if (j == 49 ):
            break

# audio
audio_thread = threading.Thread(target=f1,name="th",args=())
audio_thread.setDaemon(True)
audio_thread.start()

main_thread = threading.Thread(target=f2,name="th",args=())
main_thread.start()
