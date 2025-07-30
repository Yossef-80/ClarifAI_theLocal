import threading
import time

# Create events for synchronization
event2 = threading.Event()
event3 = threading.Event()
event4 = threading.Event()

def thread1():
    while(True):
        print("Thread 1: Sleeping for 100ms...")
        time.sleep(0.1)  # 100ms = 0.1s
        print("Thread 1: Done sleeping, triggering Thread 2")
        event2.set()

def thread2():
    while (True):
        event2.wait()  # Wait until thread1 is done
        event2.clear()  # Reset the event
        print("Thread 2: Triggered by Thread 1")
        event3.set()

def thread3():
    while(True):
        event3.wait()
        event3.clear()  # Reset the event
        print("Thread 3: Triggered by Thread 2")
        event4.set()

def thread4():
    while(True):
        event4.wait()
        event4.clear()  # Reset the event
        print("Thread 4: Triggered by Thread 3")

# Create threads
t1 = threading.Thread(target=thread1)
t2 = threading.Thread(target=thread2)
t3 = threading.Thread(target=thread3)
t4 = threading.Thread(target=thread4)

# Start threads
t1.start()
t2.start()
t3.start()
t4.start()

# Wait for all threads to complete
t1.join()
t2.join()
t3.join()
t4.join()

print("All threads completed.")
